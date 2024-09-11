import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from flask import Flask
import sqlalchemy
import dash_bootstrap_components as dbc
from waitress import serve # type: ignore

# Database connection details
db_host = 'hotel-cloud-db-dev.cy9have47g8u.eu-west-2.rds.amazonaws.com'
db_port = '5432'
db_name = 'hotelcloud'
db_user = 'hotelcloudadmin'
db_password = 'aX2X1i7z4CUUQihoSAdasd'

# Create SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Query to fetch hotel names and IDs
hotel_query = """
SELECT DISTINCT h.hotel_id, h.name
FROM booking b
JOIN hotel h ON b.hotel_id = h.hotel_id
WHERE b.hotel_id IN (5, 3, 6, 7, 4, 62)
"""
hotel_df = pd.read_sql_query(hotel_query, engine)
hotel_options = [{'label': row['name'], 'value': row['hotel_id']} for _, row in hotel_df.iterrows()]

# Query for initial data
query = """
WITH rate_updates AS (
    SELECT DISTINCT ON (u.hotel_id, date_update::date)
        u.hotel_id,
        u.rate_update_id,
        date_update::date AS report_date
    FROM rate_update u
    WHERE u.hotel_id = '6'
        AND u.date_update::date >= '2024-01-01'
    ORDER BY u.hotel_id, date_update::date, date_update DESC
),
ota_rooms AS (
    SELECT DISTINCT
        o.ota_room_id,
        COALESCE(rc."name", r."name") AS "name",
        o.room_id,
        r."name" AS original
    FROM ota_room o
    JOIN room r ON r.room_id = o.room_id
    JOIN booking b ON r.room_id = b.room_id
    LEFT JOIN room_category rc ON rc.room_category_id = r.room_category_id
    WHERE o.hotel_id = '6'
),
booking AS (
    SELECT
        b.booking_id, ROUND(COALESCE(br.total_revenue, b.total_revenue / NULLIF(b.nights, 0)) * 1.2) AS exp_rate, 
        b.hotel_id,
        b.room_id,
        COUNT(b.booking_reference) AS number_of_bookings,
        b.created_date::date AS created_date,
        b.check_in,
        b.check_out,
        b.cancel_date::date AS cancel_date,
        b.booking_reference,
        EXTRACT(DAY FROM (dt."date" - b.created_date::date)) AS date_difference,
        b.booking_channel_name,
        b.booking_status,
        b.adults,
        b.rate_plan_code,
        b.nights,
        ROUND(COALESCE(br.total_revenue, b.total_revenue / NULLIF(b.nights, 0)), 2) AS total_revenue,
        h."name" AS hotel_name,
        r."name" AS room_name,
        r.code AS room_code,
        dt."date" AS stay_date,
        p.first_name,
        p.last_name
    FROM booking b
    JOIN hotel h ON b.hotel_id = h.hotel_id
    JOIN room r ON b.room_id = r.room_id
    JOIN profile p ON p.profile_id = b.profile_id
    JOIN caldate dt ON b.check_in <= dt."date"
        AND (b.check_out > dt."date" OR (b.check_in = b.check_out AND dt."date" = b.check_in))
    LEFT JOIN booking_rate br ON b.booking_id = br.booking_rate_id
    WHERE dt."date" >= '2024-01-01' AND b.created_date >= '2024-01-01'
        AND b.hotel_id = '6'
    GROUP BY 
        b.booking_id, br.total_revenue,
        b.hotel_id, 
        b.room_id,
        b.created_date, 
        b.check_in, 
        b.check_out,
        b.cancel_date, 
        b.booking_reference,
        EXTRACT(DAY FROM (dt."date" - b.created_date::date)), 
        b.booking_channel_name,
        b.booking_status,
        b.adults,
        b.rate_plan_code,
        b.nights,
        h."name", 
        r."name", 
        r.code, 
        dt."date", 
        p.first_name, 
        p.last_name
)
SELECT
    u.hotel_id, b.room_name, b.exp_rate,
    b.rate_plan_code, 
    b.created_date, b.booking_status,
    b.booking_id, b.nights,
    u.report_date::date,
    b.booking_channel_name,
    r.stay_date::date AS stay_date,
    b.total_revenue,
    r.adultcount,
    COUNT(DISTINCT b.booking_reference) AS number_of_bookings,
    MIN(r.amount) AS refundable_rate1,
    MIN(CASE WHEN r.refundable THEN r.amount END) AS refundable_rate,
    MIN(CASE WHEN NOT r.refundable THEN r.amount END) AS non_refundable_rate,
    o.original,
    b.cancel_date,
    o.name,
    b.booking_reference,
    b.check_in,
    b.check_out
FROM rate_new r
JOIN rate_updates u ON r.rate_update_id = u.rate_update_id
JOIN ota_rooms o ON o.ota_room_id = r.ota_room_id
JOIN booking b ON b.room_id = o.room_id AND b.created_date::date = u.report_date::date
JOIN caldate dt ON b.check_in <= dt."date"
    AND (b.check_out > dt."date" OR (b.check_in = b.check_out AND dt."date" = b.check_in))
    AND dt."date" = r.stay_date::date
WHERE dt.date = r.stay_date::date AND r.stay_date >= '2024-01-01'
    AND b.adults = r.adultcount
GROUP BY
    u.hotel_id, b.room_name, b.exp_rate,
    b.rate_plan_code, b.booking_status,
    b.created_date,  
    b.booking_id, b.nights,
    u.report_date,
    r.stay_date::date,
    b.total_revenue,
    r.adultcount,
    o.name,
    b.booking_channel_name,
    b.cancel_date,
    o.original,
    b.booking_reference,
    b.check_in,
    b.check_out
"""

# Fetch initial data
df = pd.read_sql_query(query, engine)

# Close the connection
engine.dispose()

# Create a Flask server instance
server = Flask(__name__)

# Create the Dash app
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Adjusted custom color scale to ensure 0.0 is white and only starts transitioning at a higher point
# Create a custom color scale to enhance color differentiation
custom_colorscale = [
    [0, 'white'],
    [0.0001, 'white'],
    [0.1, 'yellow'],
    [0.4, 'blue'],
    [0.6, 'orange'],
    [0.8, 'red'],
    [1, 'brown']
]


def create_heatmaps(df, booking_title, revenue_title, rate_title, colorscale):
    df.fillna({'booking_channel_name': 'Unknown'}, inplace=True)
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['stay_date'] = pd.to_datetime(df['stay_date'])

    # Generate a complete date range for the booking dates
    complete_date_range = pd.date_range(start=df['created_date'].min(), end=df['created_date'].max())
    complete_date_range_str = complete_date_range.strftime('%Y-%m-%d')

    # Aggregating data
    df_agg = df.groupby(['created_date', 'stay_date', 'booking_channel_name', 'rate_plan_code', 'exp_rate']).agg({
        'number_of_bookings': 'sum',
        'total_revenue': 'sum',
        'refundable_rate': 'min',
        'non_refundable_rate': 'min'
    }).reset_index()

    # Calculate the maximum rate between refundable and non-refundable rates
    df_agg['refundable_rate1'] = df_agg[['refundable_rate', 'non_refundable_rate']].max(axis=1)

    # Convert dates to strings for categorical handling
    df_agg['created_date_str'] = df_agg['created_date'].dt.strftime('%Y-%m-%d')
    df_agg['stay_date_str'] = df_agg['stay_date'].dt.strftime('%Y-%m-%d')

    # Pivot tables for bookings, revenue, and refundable rates
    bookings_pivot = df_agg.pivot_table(
        index="stay_date_str",  # Switch index and columns
        columns="created_date_str",
        values="number_of_bookings",
        fill_value=0,
        aggfunc='sum'
    )
    
    revenue_pivot = df_agg.pivot_table(
        index="stay_date_str",  # Switch index and columns
        columns="created_date_str",
        values="total_revenue",
        fill_value=0,
        aggfunc='sum'
    )
    
    # Pivot table for the maximum refundable or non-refundable rate
    refundable_pivot = df_agg.pivot_table(
        index="stay_date_str",  # Switch index and columns
        columns="created_date_str",
        values="refundable_rate1",
        fill_value=0,
        aggfunc='min'
    )

    # Align all data to ensure the same shape
    refundable_data = refundable_pivot.reindex(index=bookings_pivot.index, columns=bookings_pivot.columns, fill_value=0)
    customdata_revenue = revenue_pivot.reindex(index=bookings_pivot.index, columns=bookings_pivot.columns, fill_value=0).values
    channel_names = df_agg.groupby(['stay_date_str', 'created_date_str'])['booking_channel_name'].apply(lambda x: ', '.join(x.unique())).unstack().reindex(index=bookings_pivot.index, columns=bookings_pivot.columns, fill_value='').values

    combined_customdata = np.dstack((
        channel_names,  
        customdata_revenue,  
        refundable_data.values,
        df_agg.pivot_table(
            index="stay_date_str",  # Switch index and columns
            columns="created_date_str",
            values="refundable_rate",
            fill_value=0,
            aggfunc='min'
        ).reindex(index=bookings_pivot.index, columns=bookings_pivot.columns, fill_value=0).values,
        df_agg.pivot_table(
            index="stay_date_str",  # Switch index and columns
            columns="created_date_str",
            values="non_refundable_rate",
            fill_value=0,
            aggfunc='min'
        ).reindex(index=bookings_pivot.index, columns=bookings_pivot.columns, fill_value=0).values
    ))

    bookings_max = bookings_pivot.values.max()
    revenue_max = revenue_pivot.values.max()
    max_rate_value = refundable_data.values.max()

    booking_fig = go.Figure(data=go.Heatmap(
        z=bookings_pivot.values,
        x=bookings_pivot.columns,
        y=bookings_pivot.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Booking Date: %{x}<br>' +
            'Stay Date: %{y}<br>' +
            'Total Number of Bookings: %{z}<br>' +
            'Total Revenue: %{customdata[1]:.2f}<br>'
        ),
        colorscale=colorscale,
        colorbar=dict(
            title="Number of Bookings",
            orientation='h',
            x=0.5,
            y=-0.3,
            len=0.6,
            thickness=15
        ),
        zmin=0,
        zmax=bookings_max
    ))

    revenue_fig = go.Figure(data=go.Heatmap(
        z=revenue_pivot.values,
        x=revenue_pivot.columns,
        y=revenue_pivot.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Booking Date: %{x}<br>' +
            'Stay Date: %{y}<br>' +
            'Total Revenue: %{customdata[1]:.2f}<br>'
        ),
        colorscale=colorscale,
        colorbar=dict(
            title="Total Revenue",
            orientation='h',
            x=0.5,
            y=-0.3,
            len=0.6,
            thickness=15
        ),
        zmin=0,
        zmax=revenue_max
    ))

    # Initialize arrays for highlighting
    highlight_refundable = np.zeros_like(refundable_data.values)  # For plus signs
    highlight_non_refundable = np.zeros_like(refundable_data.values)  # For plus signs

    for _, row in df_agg.iterrows():
        stay_date = row['stay_date_str']
        booking_date = row['created_date_str']     
    # Ensure both stay_date and booking_date are in the pivot table's index and columns
        if row['rate_plan_code'] == 'FLRA1':
                if row['refundable_rate'] == row['exp_rate']:
                    highlight_refundable[refundable_data.index.get_loc(stay_date), refundable_data.columns.get_loc(booking_date)] = 1
                if row['non_refundable_rate'] == row['exp_rate']:
                    highlight_non_refundable[refundable_data.index.get_loc(stay_date), refundable_data.columns.get_loc(booking_date)] = 1


    # Create rate heatmap with borders for highlight
    rate_fig = go.Figure()

    # Base heatmap
    rate_fig.add_trace(go.Heatmap(
        z=refundable_data.values,
        x=bookings_pivot.columns,
        y=bookings_pivot.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Stay Date: %{x}<br>' +
            'Booking Date: %{y}<br>' +
            'Refundable Rate: %{customdata[3]:.2f}<br>' +
            'Non-Refundable Rate: %{customdata[4]:.2f}<br>'
        ),
        colorscale=colorscale,
        colorbar=dict(
            title="Rate",
            orientation='h',
            x=0.5,
            y=-0.3,
            len=0.6,
            thickness=15
        ),
        zmin=0,
        zmax=max_rate_value
    ))

# Initialize the text array for highlights
    highlight_text = np.empty_like(refundable_data.values, dtype=str)

    # Fill in plus and minus signs where there are highlights
    highlight_text[highlight_refundable == 1] = 'x'
    highlight_text[highlight_non_refundable == 1] = '0'

    for text, color, highlight_array in [('x', 'green', highlight_refundable), ('o', 'red', highlight_non_refundable)]:
        rate_fig.add_trace(go.Heatmap(
            z=refundable_data.values,  # Dummy z data
            x=bookings_pivot.columns,
            y=bookings_pivot.index,
            text=np.where(highlight_array == 1, text, ''),  # Apply plus or minus signs
            texttemplate='%{text}',
            colorscale=[[0, 'rgba(0, 0, 0, 0)'], [1, 'rgba(0, 0, 0, 0)']],  # Fully transparent heatmap
            showscale=False,
            hoverinfo='skip',
            textfont=dict(size=20, color=color, family='Arial')  # Color for the signs
        ))


    booking_fig.update_layout(
        title={
            'text': booking_title,
            'font': {'size': 20, 'color': 'black', 'family': 'Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Booking Date',
        yaxis_title='Stay Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,  # Adjust height if needed
        xaxis=dict(
            tickfont=dict(size=18),
            type='category',
            showgrid=False,
            categoryarray=complete_date_range_str,  # Ensure this contains the correct date range
            gridcolor='LightGray',
            gridwidth=1          
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            showticklabels=True,
            type='category',
            categoryorder='array',
            categoryarray=complete_date_range_str,  # Ensure this contains the correct date range
            showgrid=False,
            gridcolor='LightGray',
            gridwidth=1  
        ),  # Adjust margins for space, especially if colorbar is below
    )

    revenue_fig.update_layout(
        title={
            'text': revenue_title,
            'font': {'size': 20, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Booking Date',
        yaxis_title='Stay Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,
        xaxis=dict(
            tickfont=dict(size=18),
            type='category',
            showgrid=False,
            categoryarray=complete_date_range_str,
            gridcolor='LightGray',
            gridwidth=1          
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            showticklabels=True,
            type='category',
            categoryorder='array',
            categoryarray=complete_date_range_str,
            showgrid=False,
            gridcolor='LightGray',
            gridwidth=1  
        ),
    )

    rate_fig.update_layout(
        title={
            'text': rate_title,
            'font': {'size': 20, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Booking Date',
        yaxis_title='Stay Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,
        xaxis=dict(
            tickfont=dict(size=18),
            type='category',
            showgrid=False,
            categoryarray=complete_date_range_str,
            gridcolor='LightGray',
            gridwidth=1          
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            showticklabels=True,
            type='category',
            categoryorder='array',
            categoryarray=complete_date_range_str,
            showgrid=False,
            gridcolor='LightGray',
            gridwidth=1  
        ),
        showlegend=False
    )

    return booking_fig, revenue_fig, rate_fig


def fetch_booking_details(stay_date, created_date, selected_hotel, selected_channels, selected_rooms, selected_rate_plan, selected_booking_status, selected_nights):
    channel_filter = f"AND b.booking_channel_name IN ({', '.join(f'\'{channel}\'' for channel in selected_channels)})" if selected_channels else ""
    room_filter = f"AND r.name IN ({', '.join(f'\'{room}\'' for room in selected_rooms)})" if selected_rooms else ""
    rate_plan_filter = f"AND b.rate_plan_code IN ({', '.join(f'\'{rate}\'' for rate in selected_rate_plan)})" if selected_rate_plan else ""
    book_status_filter = f"AND b.booking_status IN ({', '.join(f'\'{book}\'' for book in selected_booking_status)})" if selected_booking_status else ""
    nights_filter = f"AND b.nights IN ({', '.join(f'\'{night}\'' for night in selected_nights)})" if selected_nights else ""
    detail_query = f"""
    WITH rate_updates AS (
        SELECT DISTINCT ON (u.hotel_id, date_update::date) 
            u.hotel_id, 
            u.rate_update_id, 
            date_update::date AS report_date
        FROM rate_update u
        WHERE u.hotel_id = 6 
            AND u.date_update::date = '{created_date}' 
            AND u.date_update::date < CURRENT_DATE
        ORDER BY u.hotel_id, date_update::date, date_update DESC
    ),
    ota_rooms AS (
        SELECT DISTINCT 
            o.ota_room_id, 
            COALESCE(rc."name", r."name") AS "name", 
            o.room_id, 
            r."name" AS original
        FROM ota_room o
        JOIN room r ON r.room_id = o.room_id
        JOIN booking b ON r.room_id = b.room_id 
        LEFT JOIN room_category rc ON rc.room_category_id = r.room_category_id
        WHERE o.hotel_id = 6
    ),
    booking AS (
        SELECT
            b.booking_id, ROUND(COALESCE(br.total_revenue, b.total_revenue / NULLIF(b.nights, 0)) * 1.2) AS exp_rate, 
            b.hotel_id,
            b.room_id,
            b.created_date::date AS created_date,
            b.check_in,
            b.check_out, 
            b.cancel_date::date AS cancel_date,
            b.booking_reference, 
            EXTRACT(DAY FROM (dt."date" - b.created_date::date)) AS date_difference,
            b.booking_channel_name,
            b.booking_status, 
            b.adults,
            b.rate_plan_code,
            b.nights,
            ROUND(COALESCE(br.total_revenue, b.total_revenue / NULLIF(b.nights, 0)), 2) AS total_revenue_x,
            h."name" AS hotel_name,
            r."name" AS room_name, 
            r.code AS room_code,
            dt."date" AS stay_date,
            p.first_name,
            p.last_name
        FROM booking b
        JOIN hotel h ON b.hotel_id = h.hotel_id
        JOIN room r ON b.room_id = r.room_id
        JOIN profile p ON p.profile_id = b.profile_id
        JOIN caldate dt ON b.check_in <= dt."date"
            AND (b.check_out > dt."date" OR (b.check_in = b.check_out AND dt."date" = b.check_in))
        LEFT JOIN booking_rate br ON b.booking_id = br.booking_rate_id
        WHERE b.created_date = '{created_date}'
            AND dt."date" = '{stay_date}'
            AND b.hotel_id = {selected_hotel}
                {channel_filter}
                {room_filter}
                {rate_plan_filter}
                {book_status_filter}
                {nights_filter}
    )
    SELECT 
        u.hotel_id, b.room_name, b.room_code,
        b.rate_plan_code, b.booking_status,
        u.report_date, b.created_date,
        b.booking_channel_name,
        r.stay_date, b.date_difference, 
        b.stay_date, 
        b.total_revenue_x, b.exp_rate,
        r.adultcount, 
        MIN(CASE WHEN r.refundable THEN r.amount END) AS refundable_rate,
        MIN(CASE WHEN NOT r.refundable THEN r.amount END) AS non_refundable_rate,
        o.original, 
        b.cancel_date, 
        o.name, 
        b.booking_reference, 
        b.check_in, b.booking_id,
        b.check_out 
    FROM rate_new r
    JOIN rate_updates u ON r.rate_update_id = u.rate_update_id
    JOIN ota_rooms o ON o.ota_room_id = r.ota_room_id
    JOIN booking b ON b.room_id = o.room_id AND b.created_date = u.report_date
    JOIN caldate dt ON b.check_in <= dt."date"
        AND (b.check_out > dt."date" OR (b.check_in = b.check_out AND dt."date" = b.check_in)) 
        AND dt."date" = r.stay_date
    WHERE dt.date = r.stay_date and r.stay_date = '{stay_date}'
        AND b.adults = r.adultcount
    GROUP BY 
        u.hotel_id, b.room_name, b.room_code, b.booking_id,
        u.report_date, b.booking_status,
        b.rate_plan_code, b.created_date,
        r.stay_date, 
        b.adults, 
        o.name, 
        b.booking_channel_name, b.exp_rate,
        b.total_revenue_x,
        r.adultcount, b.date_difference,
        b.stay_date, 
        b.booking_reference, 
        b.check_in, 
        b.check_out, 
        b.cancel_date, 
        o.original;
    """
    
    return pd.read_sql_query(detail_query, engine)

# Define the bar chart for booking channels
def fetch_data_with_sql_query(selected_hotel, stay_date):
    # Define the SQL query using parameterized placeholders
    sql_query = text("""
    WITH aggregated_data AS (
        SELECT
            dt."date" AS stay_date,
            h."name" AS hotel_name,
            b.created_date::date AS created_date, 
            EXTRACT(DAY FROM (dt."date" - b.created_date::date)) AS date_difference,
            h.hotel_id,
            b.booking_channel_name,
            b.booking_status, 
            r."name" AS room_name,
            b.rate_plan_code,
            COUNT(b.booking_reference) AS number_of_bookings,
            SUM(COALESCE(br.total_revenue, b.total_revenue / (CASE WHEN b.nights=0 THEN 1 ELSE b.nights END))) AS total_revenue
        FROM
            booking b
        JOIN
            hotel h ON b.hotel_id = h.hotel_id
        JOIN 
            room r ON b.room_id = r.room_id
        JOIN 
            caldate dt ON b.check_in <= dt."date" 
            AND ((b.check_out > dt."date") OR ((b.check_in = b.check_out) AND (dt."date" = b.check_in)))
        LEFT OUTER JOIN 
            booking_rate br ON b.booking_id = br.booking_rate_id
        WHERE
            dt."date" = :stay_date
            AND b.hotel_id = :selected_hotel AND b.booking_status = 'CheckedOut'
        GROUP BY
            dt."date",
            b.created_date::date,
            h.name,
            h.hotel_id,
            b.rate_plan_code,
            b.booking_status,
            b.booking_channel_name,
            r."name"
    )
    SELECT
        stay_date,
        hotel_name,
        created_date,
        hotel_id,
        booking_channel_name, 
        date_difference,
        booking_status,
        room_name,
        rate_plan_code,
        number_of_bookings,
        total_revenue,
        SUM(number_of_bookings) OVER (
            PARTITION BY stay_date, booking_status
            ORDER BY created_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_bookings
    FROM
        aggregated_data
    ORDER BY
        hotel_id, stay_date, created_date;
    """)
    
    # Execute the query using parameters
    return pd.read_sql_query(sql_query, engine, params={"stay_date": stay_date, "selected_hotel": selected_hotel})

# Layout of the Dash app
app.layout = dbc.Container([
    dcc.Store(id='last-clicked-heatmap', data='none'),

    dbc.Row([
        dbc.Col(html.H1("Hotel Booking Dashboard"), width={"size": 6, "offset": 4})
    ]),

    dbc.Row([
        dbc.Col([
            html.Div("Stay Date:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '20px', 'fontFamily': 'Arial'}),
            dcc.DatePickerRange(
                id='stay-date-picker',
                start_date='2024-01-01',
                end_date='2024-12-31',
                display_format='YYYY-MM-DD',  # Format for displaying date
                style={'width': '100%', 'padding': '10px'}
            ),
        ], width=2),

        dbc.Col([
            html.Div("Booking Date:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '20px', 'fontFamily': 'Arial'}),
            dcc.DatePickerRange(
                id='created-date-picker',
                start_date='2024-01-01',
                end_date='2024-12-31',
                display_format='YYYY-MM-DD',  # Format for displaying date
                style={'width': '100%', 'padding': '10px'}
            ),
        ], width=2),
    ], style={'marginBottom': '20px'}),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='hotel-dropdown',
                options=hotel_options,
                value=hotel_options[0]['value'],  # Default value
                style={
                    'width': '100%', 
                    'fontSize': '20px',  # Font size
                    'fontFamily': 'Arial',  # Font family
                    'color': 'black'  # Font color
                },
                clearable=False,
                placeholder='Select a hotel'
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='channel-dropdown',
                options=[],  # Initially empty
                value=[],  # Default to an empty list (no channels selected)
                multi=True,  # Enable multi-select
                style={
                    'width': '100%', 
                    'fontSize': '16px',  # Font size
                    'fontFamily': 'Arial',  # Font family
                    'color': 'black'  # Font color
                },
                placeholder='Select booking channels'
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='room-dropdown',
                multi=True,
                style={
                    'width': '100%', 
                    'fontSize': '16px',  # Font size
                    'fontFamily': 'Arial',
                    'color': 'black'  # Font color
                },
                placeholder='Select room types'
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='rate-dropdown',
                options=[],  # Initially empty
                value=[],  # Default to an empty list (no channels selected)
                multi=True,
                style={
                    'width': '100%', 
                    'fontSize': '16px',  # Font size
                    'fontFamily': 'Arial',
                    'color': 'black'  # Font color
                },
                placeholder='Select rate codes'  # Placeholder text
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='book-dropdown',
                options=[],  # Initially empty
                value=[], 
                multi=True,
                style={
                    'width': '100%', 
                    'fontSize': '16px',  # Font size
                    'fontFamily': 'Arial',  # Font family
                    'color': 'black'  # Font color
                },
                placeholder='Select Booking Status'  # Placeholder text
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id='night-dropdown',
                options=[],  # Initially empty
                value=[], 
                multi=True,
                style={
                    'width': '100%', 
                    'fontSize': '16px',  # Font size
                    'fontFamily': 'Arial',  # Font family
                    'color': 'black'  # Font color
                },
                placeholder='Select Nights'  # Placeholder text
            ),
        ], width=2),
    ], style={'marginBottom': '20px'}),

dcc.Tabs([
    dcc.Tab(label='Main Dashboard', children=[
        # Row for heatmaps
        dbc.Row([
            dbc.Button('Toggle Heatmap', id='toggle-button', n_clicks=0),
            dbc.Col(dcc.Graph(id='heatmap1', style={'height': '800px'}), width=6),  # Set fixed height
            dbc.Col(dcc.Graph(id='heatmap2', style={'height': '800px'}), width=6)
        ], style={'marginBottom': '40px'}),  # Set fixed height),
        # Add margin to this row to increase space between heatmaps and the line chart
        dbc.Row([ html.H2('Booking Trend'),
            dbc.Col(dcc.Graph(id='line-chart', style={'height': '800px'}), width=12)
        ], style={'marginTop': '40px'}),  # Set fixed height

        # Booking Details Section
        html.Div([
            html.H2('Booking Details'),
            dcc.Loading(
                id="loading-booking-details",
                type="circle",
                children=[
                    dash_table.DataTable(
                        id='booking-details',
                        columns=[
                            {'name': 'Booking Reference', 'id': 'booking_reference'},
                            {'name': 'Booking Status', 'id': 'booking_status'},
                            {'name': 'Refundable Rate', 'id': 'refundable_rate'},
                            {'name': 'Non Refundable Rate', 'id': 'non_refundable_rate'},
                            {'name': 'Expected Rate', 'id': 'exp_rate'},
                            {'name': 'Room Name', 'id': 'room_name'},
                            {'name': 'Room Code', 'id': 'room_code'},
                            {'name': 'Lead In', 'id': 'date_difference'},
                            {'name': 'Cancel Date', 'id': 'cancel_date'},
                            {'name': 'Stay Date', 'id': 'stay_date'},
                            {'name': 'Booking Date', 'id': 'created_date'},
                            {'name': 'Total Revenue', 'id': 'total_revenue_x'},
                            {'name': 'Booking Channel', 'id': 'booking_channel_name'},
                            {'name': 'Check-in Date', 'id': 'check_in'},
                            {'name': 'Check-out Date', 'id': 'check_out'},
                            {'name': 'Rate Plan', 'id': 'rate_plan_code'},
                        ],
                        style_table={'overflowX': 'auto', 'fontSize': 14},
                        style_cell={'textAlign': 'left', 'padding': '5px', 'padding-right': '10px', 'fontSize': '18px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold', 'fontSize': 18, 'padding-right': '10px'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)',
                            },
                        ],
                        style_as_list_view=True
                    )
                ]
            )
        ], style={'width': '100%', 'padding': '10px'}),

        # Additional Booking Details Section
        html.Div([
            html.H2('Additional Booking Details'),
            dcc.Loading(
                id="loading-additional-details",
                type="circle",
                children=[
                    dash_table.DataTable(
                        id='additional-details',
                        columns=[
                            {'name': 'First Name', 'id': 'first_name'},
                            {'name': 'Last Name', 'id': 'last_name'},
                            {'name': 'Room Number', 'id': 'room_number'},
                            {'name': 'Email', 'id': 'email'},
                            {'name': 'Market Code', 'id': 'market_code'},
                        ],
                        style_table={'overflowX': 'auto', 'fontSize': 14},
                        style_cell={'textAlign': 'left', 'padding': '5px', 'padding-right': '10px', 'fontSize': '18px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold', 'fontSize': 18, 'padding-right': '10px'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)',
                            },
                        ],
                        style_as_list_view=True
                    )
                ]
            )
        ], style={'width': '100%', 'padding': '10px'}),

        # Bar Chart Section
        dbc.Row([ html.H2('Revenue Trend'),
            dbc.Col(dcc.Graph(id='bar-chart', style={'height': '600px'}), width=12)
        ], style={'marginTop': '40px'}),  # Set fixed height

        html.Div(id='hover-data', style={'display': 'none'})  # Placeholder for hover data
    ],
        style={'fontSize': '20px', 'fontFamily': 'Arial'}
    ),

        dcc.Tab(
    label='Pickup Curve', 
    children=[
        dbc.Row([
            dbc.Col([
                html.Div("Select Stay Date for Line Chart:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '20px', 'fontFamily': 'Arial'}),
                dcc.DatePickerSingle(
                    id='line-chart-stay-date-picker',
                    date='2024-01-01',  # Default date
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'padding': '10px'}
                )
            ], width=3),

            dbc.Col(dcc.Graph(id='new-line-chart', style={'height': '600px'}), width=12),  # Set fixed height
        ])
    ],
    style={'fontSize': '20px', 'fontFamily': 'Arial'}  # Set font size and family for the tab label
),
dcc.Tab(
    label='Market Competitors', 
    children=[
        dbc.Row([
        dbc.Button('Toggle Heatmap', id='toggle-button1', n_clicks=0),
        dcc.Graph(id='heatmap3'),
        dcc.Graph(id='heatmap4')  # Set fixed height
        ])
    ],
    style={'fontSize': '20px', 'fontFamily': 'Arial'}  # Set font size and family for the tab label
)
])
], fluid=True)


@app.callback(
    [Output('heatmap1', 'figure'),
     Output('heatmap2', 'figure'),
     Output('booking-details', 'data'),
     Output('bar-chart', 'figure'),
     Output('line-chart', 'figure'),  # New line chart output
     Output('channel-dropdown', 'options'),
     Output('additional-details', 'data'),
     Output('room-dropdown', 'options'),
     Output('rate-dropdown', 'options'),
     Output('book-dropdown', 'options'),
     Output('night-dropdown', 'options')],
    [Input('hotel-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('room-dropdown', 'value'),
     Input('booking-details', 'active_cell'),
     Input('booking-details', 'data'),
     Input('rate-dropdown', 'value'),
     Input('book-dropdown', 'value'),
     Input('stay-date-picker', 'start_date'),
     Input('stay-date-picker', 'end_date'),
     Input('created-date-picker', 'start_date'),
     Input('created-date-picker', 'end_date'),
     Input('toggle-button', 'n_clicks'),
     Input('heatmap1', 'relayoutData'),
     Input('heatmap2', 'relayoutData'),
     Input('heatmap3', 'relayoutData'),
     Input('heatmap1', 'clickData'),
     Input('heatmap2', 'clickData'),
     Input('heatmap3', 'clickData'),
     Input('night-dropdown', 'value')]
)
def update_output(selected_hotel, selected_channels, selected_rooms, active_cell, table_data, selected_rate_plan, selected_booking_status, stay_date_start, stay_date_end, created_date_start, created_date_end, n_clicks, booking_relayout, revenue_relayout, rate_relayout, booking_click_data, revenue_click_data, rate_click_data, selected_nights):
    # Default values
    booking_heatmap = go.Figure()
    rate_heatmap = go.Figure()
    line_chart_fig = go.Figure()
    revenue_heatmap = go.Figure()
    booking_details_data = []
    bar_chart_fig = go.Figure()
    channel_options = []
    additional_data = []
    room_options = []
    rate_options = []
    book_options = []
    nights_options = []
    # Convert date strings to datetime.date objects
    if stay_date_start:
        stay_date_start = datetime.strptime(stay_date_start, '%Y-%m-%d').date()
    if stay_date_end:
        stay_date_end = datetime.strptime(stay_date_end, '%Y-%m-%d').date()
    if created_date_start:
        created_date_start = datetime.strptime(created_date_start, '%Y-%m-%d').date()
    if created_date_end:
        created_date_end = datetime.strptime(created_date_end, '%Y-%m-%d').date()

    # Ensure dataframe date columns are in datetime format
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['stay_date'] = pd.to_datetime(df['stay_date'])

    # Convert to datetime.date for filtering
    df['created_date'] = df['created_date'].dt.date
    df['stay_date'] = df['stay_date'].dt.date

    # Filter data based on selected hotel
    filtered_df = df[df['hotel_id'] == selected_hotel] if selected_hotel else df.copy()

    # Apply date filters
    if stay_date_start and stay_date_end:
        filtered_df = filtered_df[(filtered_df['stay_date'] >= stay_date_start) & (filtered_df['stay_date'] <= stay_date_end)]
    if created_date_start and created_date_end:
        filtered_df = filtered_df[(filtered_df['created_date'] >= created_date_start) & (filtered_df['created_date'] <= created_date_end)]

    # Update dropdown options
    if 'booking_channel_name' in filtered_df.columns and not filtered_df['booking_channel_name'].isnull().all():
        channels = filtered_df['booking_channel_name'].dropna().unique()
        channel_options = [{'label': channel, 'value': channel} for channel in channels]

    if selected_channels:
        filtered_df = filtered_df[filtered_df['booking_channel_name'].isin(selected_channels)]

    if 'rate_plan_code' in filtered_df.columns:
        rate_code = filtered_df['rate_plan_code'].dropna().unique()
        sorted_rate_code = sorted([rate for rate in rate_code if rate])
        rate_options = [{'label': rate, 'value': rate} for rate in sorted_rate_code]

    if selected_rate_plan:
        filtered_df = filtered_df[filtered_df['rate_plan_code'].isin(selected_rate_plan)]

    if 'room_name' in filtered_df.columns:
        rooms = filtered_df['room_name'].dropna().unique()
        room_options = [{'label': room, 'value': room} for room in rooms]

    if selected_rooms:
        filtered_df = filtered_df[filtered_df['room_name'].isin(selected_rooms)]

    if 'booking_status' in filtered_df.columns and not filtered_df['booking_status'].isnull().all():
        bookingstatus = filtered_df['booking_status'].dropna().unique()
        book_options = [{'label': book, 'value': book} for book in bookingstatus]

    if selected_booking_status:
        filtered_df = filtered_df[filtered_df['booking_status'].isin(selected_booking_status)]

    if 'nights' in filtered_df.columns:
        nights = filtered_df['nights'].dropna().unique()
        sorted_night_code = sorted([night for night in nights if night])
        nights_options = [{'label': night, 'value': night} for night in sorted_night_code]

    if selected_nights:
        filtered_df = filtered_df[filtered_df['nights'].isin(selected_nights)]

    # Define custom colorscale for heatmaps
    custom_colorscale = [
        [0, 'white'],       # Explicitly set 0.0 to white
        [0.0001, 'white'],  # Ensure small values still map to white
        [0.1, 'yellow'],    # Start transitioning from yellow at a higher value
        [0.4, 'blue'],
        [0.6, 'orange'],
        [0.8, 'red'],
        [1, 'brown']
    ]

    # Rest of your code to generate heatmaps and charts...

    # Default to 0 clicks if n_clicks is None
    if n_clicks is None:
        n_clicks = 0

    booking_title = 'Booking Heatmap'.format(selected_hotel)
    revenue_title = 'Revenue Heatmap'.format(selected_hotel)
    rate_title = 'Rate Heatmap'.format(selected_hotel)
    
    # Toggle logic
    if n_clicks % 2 == 0:
        booking_heatmap, revenue_heatmap, rate_heatmap = create_heatmaps(filtered_df, booking_title, revenue_title, rate_title, custom_colorscale)
        fig1 = booking_heatmap
        fig2 = revenue_heatmap
    else:
        booking_heatmap, revenue_heatmap, rate_heatmap = create_heatmaps(filtered_df, booking_title, revenue_title, rate_title, custom_colorscale)
        fig1 = booking_heatmap
        fig2 = rate_heatmap

    # Synchronize heatmap zoom and pan
    x_range = y_range = None

    if booking_relayout and 'xaxis.range[0]' in booking_relayout and 'xaxis.range[1]' in booking_relayout:
        x_range = [booking_relayout['xaxis.range[0]'], booking_relayout['xaxis.range[1]']]
        y_range = [booking_relayout['yaxis.range[0]'], booking_relayout['yaxis.range[1]']]
    elif revenue_relayout and 'xaxis.range[0]' in revenue_relayout and 'xaxis.range[1]' in revenue_relayout:
        x_range = [revenue_relayout['xaxis.range[0]'], revenue_relayout['xaxis.range[1]']]
        y_range = [revenue_relayout['yaxis.range[0]'], revenue_relayout['yaxis.range[1]']]
    elif rate_relayout and 'xaxis.range[0]' in rate_relayout and 'xaxis.range[1]' in rate_relayout:
        x_range = [rate_relayout['xaxis.range[0]'], rate_relayout['xaxis.range[1]']]
        y_range = [rate_relayout['yaxis.range[0]'], rate_relayout['yaxis.range[1]']]
    
    if x_range and y_range:
        fig1.update_xaxes(range=x_range)
        fig1.update_yaxes(range=y_range)
        fig2.update_xaxes(range=x_range)
        fig2.update_yaxes(range=y_range)

    # Handle marker synchronization
    marker_data = None
    if booking_click_data:
        stay_date = booking_click_data['points'][0]['x']
        created_date = booking_click_data['points'][0]['y']
        marker_data = {'stay_date': stay_date, 'created_date': created_date}
    elif revenue_click_data:
        stay_date = revenue_click_data['points'][0]['x']
        created_date = revenue_click_data['points'][0]['y']
        marker_data = {'stay_date': stay_date, 'created_date': created_date}
    elif rate_click_data:
        stay_date = rate_click_data['points'][0]['x']
        created_date = rate_click_data['points'][0]['y']
        marker_data = {'stay_date': stay_date, 'created_date': created_date}

    if marker_data:
        stay_date = marker_data['stay_date']
        created_date = marker_data['created_date']

        # Update markers on booking heatmap
        fig1.add_trace(go.Scatter(
            x=[stay_date],
            y=[created_date],
            mode='markers',
            marker=dict(
                color='black',
                size=15,
                symbol='x',
                line=dict(color='red', width=2)
            ),
            showlegend=False
        ))

        # Update markers on revenue heatmap
        fig2.add_trace(go.Scatter(
            x=[stay_date],
            y=[created_date],
            mode='markers',
            marker=dict(
                color='black',
                size=15,
                symbol='x',
                line=dict(color='red', width=2)
            ),
            showlegend=False
        ))

        stay_date_filtered_df = filtered_df[filtered_df['stay_date'] == stay_date]

# Line Chart Data
        line_chart_data = stay_date_filtered_df.groupby('created_date').size().reset_index(name='number_of_bookings')
        line_chart_data['created_date'] = pd.to_datetime(line_chart_data['created_date'])

        # Generate a complete date range
        if not line_chart_data.empty:
            min_date = line_chart_data['created_date'].min()
            max_date = line_chart_data['created_date'].max()
        else:
            min_date = pd.to_datetime(stay_date)  # Assuming stay_date is a string or date
            max_date = min_date

        complete_date_range = pd.date_range(start=min_date, end=max_date).date
        complete_line_chart_data = pd.DataFrame(complete_date_range, columns=['created_date'])
        complete_line_chart_data['created_date'] = pd.to_datetime(complete_line_chart_data['created_date'])

        # Merge to fill missing dates
        complete_line_chart_data = pd.merge(
            complete_line_chart_data, 
            line_chart_data, 
            on='created_date', 
            how='left'
        ).fillna(0)
        
        complete_line_chart_data['number_of_bookings'] = complete_line_chart_data['number_of_bookings'].astype(int)

        # Calculate total bookings
        total_bookings = complete_line_chart_data['number_of_bookings'].sum()

        # Create Line Chart
        line_chart_fig = go.Figure()
        line_chart_fig.add_trace(go.Scatter(
            x=complete_line_chart_data['created_date'],
            y=complete_line_chart_data['number_of_bookings'],
            mode='lines+markers+text',
            line=dict(color='blue'),
            marker=dict(size=10),
            text=complete_line_chart_data['number_of_bookings'],
            textposition="top center",
            name=f'Bookings for Stay Date: {stay_date}'
        ))

        line_chart_fig.update_layout(
            title={
                'text': f'<b>Booking Trends for Stay Date: {stay_date} (Total Bookings: {total_bookings})</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'black'},
            },
            xaxis_title='Created Date',
            yaxis_title='Number of Bookings',
            xaxis=dict(
                tickformat="%b %d",
                tickangle=45,
                title_font=dict(size=20),
                tickfont=dict(size=12),
                type='category'
            ),
            yaxis=dict(
                tickformat=",.0f",
                title_font=dict(size=20),
                tickfont=dict(size=16)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            height=800,
        )
                
        # Filter data for the selected stay date and ensure created_date is in datetime format
        stay_date_filtered_df = filtered_df[filtered_df['stay_date'] == stay_date]
        stay_date_filtered_df['created_date'] = pd.to_datetime(stay_date_filtered_df['created_date'])

        # Generate a complete date range from the minimum to the maximum created date in the filtered DataFrame
        if not stay_date_filtered_df.empty:
            min_date = stay_date_filtered_df['created_date'].min()
            max_date = stay_date_filtered_df['created_date'].max()
        else:
            min_date = created_date
            max_date = created_date

        complete_date_range = pd.date_range(start=min_date, end=max_date)

        # Create DataFrame for the complete date range
        complete_revenue_data = pd.DataFrame(complete_date_range, columns=['created_date'])

        # Define color scheme
        color_scheme = ['yellow', 'blue', 'green', 'red', 'orange', 'violet', 'brown']

        def get_colors(room_names, color_scheme):
            num_colors = len(color_scheme)
            colors = [color_scheme[i % num_colors] for i in range(len(room_names))]
            return colors

        # Extract unique room names for color assignment
        room_names = stay_date_filtered_df['room_name'].unique()
        color_scale = get_colors(room_names, color_scheme)

        # Create Bar Chart
        bar_chart_fig = go.Figure()

        # Iterate through each room and add bar traces for each booking individually
        for i, room_name in enumerate(room_names):
            room_data = stay_date_filtered_df[stay_date_filtered_df['room_name'] == room_name]
            
            # Merge complete date range with room data to include all dates
            complete_room_data = pd.merge(
                complete_revenue_data,
                room_data,
                on='created_date',
                how='left'
            ).fillna({'total_revenue': 0})  # Fill missing revenues with 0
            
            # Iterate over each date to add individual segments
            bar_chart_fig.add_trace(go.Bar(
                x=complete_room_data['created_date'],
                y=complete_room_data['total_revenue'],
                marker=dict(
                    color=color_scale[i % len(color_scale)],
                    line=dict(color='black', width=1)  # Black borders with width 1
                ),
                name=room_name,  # Simplify legend entry to just the room name
                text=[f"£{revenue:.2f}<br>Bookings: {bookings}" for revenue, bookings in zip(complete_room_data['total_revenue'], complete_room_data['number_of_bookings'])],  # Show revenue and bookings
                textposition='inside',
                textfont=dict(size=20)  # Set font size for revenue labels
            ))


        bar_chart_fig.update_layout(
            barmode='stack',  # Stacked bar mode to show individual booking revenues
            title={
                'text': f'<b>Revenue Breakdown by Room for Stay Date: {stay_date}</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'black'},
                'y': 0.95,
            },
            xaxis_title='Created Date',
            yaxis_title='Revenue',
            xaxis=dict(
                tickformat="%b %d",
                tickangle=45,
                title_font=dict(size=20),
                tickfont=dict(size=12),
                type='category'
            ),
            yaxis=dict(
                tickformat="$,.2f",
                title_font=dict(size=20),
                tickfont=dict(size=16)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            height=800,
            legend=dict(
            font=dict(size=16),
            ))

      # Fetch booking details
        booking_details_df = fetch_booking_details(
            stay_date, created_date, selected_hotel, 
            selected_channels if selected_channels else [], 
            selected_rooms if selected_rooms else [], 
            selected_rate_plan if selected_rate_plan else [], 
            selected_booking_status if selected_booking_status else [],
            selected_nights if selected_nights else []
        )
        booking_details_data = booking_details_df.to_dict('records')

    # Fetch additional data from booking table click
    booking_reference = None
    if active_cell and table_data:
        row = active_cell.get('row', None)
        if row is not None and 0 <= row < len(table_data):
            booking_reference = table_data[row].get('booking_reference')
            if booking_reference:
                query = """
                SELECT p.first_name, p.last_name, b.room_number, b.hotel_id
                FROM booking b
                JOIN profile p ON b.profile_id = p.profile_id 
                WHERE b.booking_reference = :booking_reference
                """
                
                with engine.connect() as connection:
                    result = connection.execute(
                        sqlalchemy.text(query), 
                        {'booking_reference': booking_reference}
                    ).fetchall()

                if result:
                    additional_data = [
                        {'first_name': row[0], 'last_name': row[1], 'room_number': row[2], 'hotel_id': row[3]}
                        for row in result
                    ]
        
    return fig1, fig2, booking_details_data, bar_chart_fig, line_chart_fig, channel_options, additional_data, room_options, rate_options, book_options, nights_options

@app.callback(
    Output('new-line-chart', 'figure'),
    [Input('hotel-dropdown', 'value'),
     Input('line-chart-stay-date-picker', 'date'),
     Input('channel-dropdown', 'value'),
     Input('rate-dropdown', 'value')]  # Add rate dropdown input
)
def update_new_line_chart(selected_hotel, selected_stay_date, selected_channels, selected_rate_plan):
    # Check if stay date and hotel are selected
    if not selected_stay_date or not selected_hotel:
        return go.Figure()

    # Fetch data using the provided SQL query function
    new_line_chart_data = fetch_data_with_sql_query(selected_hotel, selected_stay_date)

    # Filter data based on selected stay date and hotel
    filtered_data = new_line_chart_data[
        (new_line_chart_data['stay_date'] == selected_stay_date) &
        (new_line_chart_data['hotel_id'] == selected_hotel)
    ]

    # Further filter data based on selected channels and rate plan if any
    if selected_channels:
        filtered_data = filtered_data[filtered_data['booking_channel_name'].isin(selected_channels)]
    
    if selected_rate_plan:
        filtered_data = filtered_data[filtered_data['rate_plan_code'].isin(selected_rate_plan)]

    # Initialize the figure
    new_line_chart_fig = go.Figure()

    # Define axis labels based on selected filters
    x_axis_label = 'Lead In'
    y_axis_label = 'Cumulative Bookings'
    
    if selected_channels:
        x_axis_label = 'Date Difference'  # Example, adjust based on context
        y_axis_label = 'Bookings by Channel'  # Example, adjust based on context
    
    if selected_rate_plan:
        x_axis_label = 'Rate Plan'  # Example, adjust based on context
        y_axis_label = 'Bookings by Rate Plan'  # Example, adjust based on context

    # If channels are selected, plot separate lines for each channel
    if selected_channels:
        for channel in selected_channels:
            channel_data = filtered_data[filtered_data['booking_channel_name'] == channel]
            new_line_chart_fig.add_trace(go.Scatter(
                x=channel_data['date_difference'],
                y=channel_data['cumulative_bookings'],
                mode='lines+markers',
                line=dict(width=2, shape='spline'),
                fill='tozeroy',
                marker=dict(size=10),
                name=channel  # Channel name for legend
            ))
    else:
        # Plot a single line if no channels are selected
        new_line_chart_fig.add_trace(go.Scatter(
            x=filtered_data['date_difference'],
            y=filtered_data['cumulative_bookings'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='blue', shape='spline'),
            marker=dict(size=10),
            name='Overall Bookings'
        ))

    new_line_chart_fig.update_layout(
        title={
            'text': f'Pickup Curve for Stay Date {selected_stay_date}',
            'font': {'size': 20, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        template='plotly_white',
        xaxis=dict(
            tickformat="%d",
            tickangle=45,
            title_font=dict(size=18),  # Font size for x-axis title
            tickfont=dict(size=18),    # Font size for x-axis labels
            autorange='reversed'  # Reverse the x-axis
        ),
        yaxis=dict(
            tickformat=",.0f",
            title_font=dict(size=18),  # Font size for y-axis title
            tickfont=dict(size=18)     # Font size for y-axis labels
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        height=1000,
    )

    return new_line_chart_fig

if __name__ == '__main__':
    serve(app.server, host='0.0.0.0', port=8050)
