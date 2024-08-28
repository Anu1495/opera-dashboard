import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
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
SELECT
    dt."date" AS stay_date,
    h."name",
    b.created_date::date AS created_date,
    h.hotel_id,
    b.booking_channel_name,
    b.booking_status, 
    r."name" AS room_name,
    b.rate_plan_code,  -- Include room_name in the query
    COUNT(b.booking_reference) AS number_of_bookings,
    SUM(COALESCE(br.total_revenue, b.total_revenue / (CASE WHEN b.nights=0 THEN 1 ELSE b.nights END))) AS total_revenue
FROM
    booking b
JOIN
    hotel h ON b.hotel_id = h.hotel_id
JOIN 
        room r ON b.room_id=r.room_id
JOIN 
    caldate dt ON b.check_in <= dt."date" 
    AND ((b.check_out > dt."date") OR ((b.check_in = b.check_out) AND (dt."date" = b.check_in)))
LEFT OUTER JOIN 
    booking_rate br ON b.booking_id=br.booking_rate_id
WHERE
    dt."date" >= '2024-01-01' AND b.created_date >= '2024-01-01' 
    AND b.hotel_id IN (5, 3, 6, 7, 4, 62)
GROUP BY
    dt."date",
    b.created_date::date,
    h.name,
    h.hotel_id,
    b.rate_plan_code,
    b.booking_status,
    b.booking_channel_name,
    r."name" 
    
ORDER BY
    b.created_date::date,
    dt."date";
"""

# Fetch initial data
df = pd.read_sql_query(query, engine)

# Close the connection
engine.dispose()

# Create a Flask server instance
server = Flask(__name__)



# Create the Dash app
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the custom color scale
custom_colorscale = [
    [0, 'white'],
    [0.1, 'yellow'],
    [0.4, 'blue'],
    [0.6, 'orange'],
    [0.8, 'red'],
    [1, 'brown']
]

def create_heatmaps(df, booking_title, revenue_title, colorscale):
    df.fillna({'booking_channel_name': 'Unknown'}, inplace=True)
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['stay_date'] = pd.to_datetime(df['stay_date'])

    # Generate a complete date range for the booking dates
    complete_date_range = pd.date_range(start=df['created_date'].min(), end=df['created_date'].max())
    complete_date_range_str = complete_date_range.strftime('%Y-%m-%d')

    # Aggregating data
    df_agg = df.groupby(['created_date', 'stay_date', 'booking_channel_name']).agg({
        'number_of_bookings': 'sum',
        'total_revenue': 'sum'
    }).reset_index()

    # Converting dates to strings for categorical handling
    df_agg['created_date_str'] = df_agg['created_date'].dt.strftime('%Y-%m-%d')
    df_agg['stay_date_str'] = df_agg['stay_date'].dt.strftime('%Y-%m-%d')

    # Pivot tables for bookings and revenue
    bookings_pivot = df_agg.pivot_table(index="created_date_str", columns="stay_date_str", values="number_of_bookings", fill_value=0, aggfunc='sum')
    revenue_pivot = df_agg.pivot_table(index="created_date_str", columns="stay_date_str", values="total_revenue", fill_value=0, aggfunc='sum')

    customdata_revenue = revenue_pivot.reindex(complete_date_range_str, fill_value=0).values
    channel_names = df_agg.groupby(['created_date_str', 'stay_date_str'])['booking_channel_name'].apply(lambda x: ', '.join(x.unique())).unstack().reindex(complete_date_range_str, fill_value='').values

    # Combining custom data
    combined_customdata = np.dstack((channel_names, customdata_revenue))

    # Creating the booking heatmap
    booking_fig = go.Figure(data=go.Heatmap(
        z=bookings_pivot.values,
        x=bookings_pivot.columns,
        y=bookings_pivot.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Booking Date: %{y}<br>' +
            'Stay Date: %{x}<br>' +
            'Total Number of Bookings: %{z}<br>' +
            'Total Revenue: %{customdata[1]:.2f}'
        ),
        colorscale=colorscale,
        colorbar=dict(title="Number of Bookings", tickvals=[0, 5, 10], ticktext=['0', '5', '10']),
        zmin=0,
        zmax=10
    ))

    # Creating the revenue heatmap
    revenue_fig = go.Figure(data=go.Heatmap(
        z=revenue_pivot.values,
        x=revenue_pivot.columns,
        y=revenue_pivot.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Booking Date: %{y}<br>' +
            'Stay Date: %{x}<br>' +
            'Total Revenue: %{z:.2f}<br>'
        ),
        colorscale=colorscale,
        colorbar=dict(title="Total Revenue", tickvals=[0, 1000, 2000], ticktext=['0', '1000', '2000']),
        zmin=0,
        zmax=2000
    ))

    # Updating layout for both heatmaps
    booking_fig.update_layout(
        title={
            'text': booking_title,
            'font': {'size': 20, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Stay Date',
        yaxis_title='Booking Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        xaxis=dict(
            tickfont=dict(size=18),
            type='category',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1          
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            showticklabels=True,
            type='category',
            categoryorder='array',
            categoryarray=complete_date_range_str,
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1  
        ),
    )

    revenue_fig.update_layout(
        title={
            'text': revenue_title,
            'font': {'size': 20, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Stay Date',
        yaxis_title='Booking Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        xaxis=dict(
            tickfont=dict(size=18),
            type='category',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1          
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            showticklabels=True,
            type='category',
            categoryorder='array',
            categoryarray=complete_date_range_str,
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1  
        ),
    )

    return booking_fig, revenue_fig



def fetch_booking_details(stay_date, created_date, selected_hotel, selected_channels, selected_rooms, selected_rate_plan, selected_booking_status):
    channel_filter = f"AND b.booking_channel_name IN ({', '.join(f'\'{channel}\'' for channel in selected_channels)})" if selected_channels else ""
    room_filter = f"AND r.name IN ({', '.join(f'\'{room}\'' for room in selected_rooms)})" if selected_rooms else ""
    rate_plan_filter = f"AND b.rate_plan_code IN ({', '.join(f'\'{rate}\'' for rate in selected_rate_plan)})" if selected_rate_plan else ""
    book_status_filter = f"AND b.booking_status IN ({', '.join(f'\'{book}\'' for book in selected_booking_status)})" if selected_booking_status else ""
    detail_query = f"""
    SELECT
        dt."date"::date AS stay_date,
        b.created_date::date,
        b.cancel_date::date,
        b.booking_reference,
        ROUND(COALESCE(br.total_revenue, b.total_revenue / (CASE WHEN b.nights=0 THEN 1 ELSE b.nights END)), 2) AS total_revenue,
        COUNT(b.booking_reference) AS number_of_bookings,
        COUNT(b.booking_reference) AS number_of_bookings,
        b.booking_status,
        EXTRACT(DAY FROM (dt."date" - b.created_date::date)) AS date_difference,
        b.booking_channel_name,
        r."name" AS room_name,
        b.room_number,
        b.check_in,
        b.check_out,
        b.rate_plan_code,
        h."name", r.code,
        b.nights
    FROM
        booking b
    JOIN
        hotel h ON b.hotel_id = h.hotel_id
    JOIN 
        room r ON b.room_id=r.room_id
    JOIN 
        caldate dt ON b.check_in <= dt."date" 
        AND ((b.check_out > dt."date") OR ((b.check_in = b.check_out) AND (dt."date" = b.check_in)))
    LEFT OUTER JOIN 
        booking_rate br ON b.booking_id=br.booking_rate_id

    WHERE
        dt."date"::date = '{stay_date}'
        AND b.created_date = '{created_date}'
        AND b.hotel_id = {selected_hotel}
        {channel_filter}
        {room_filter}
        {rate_plan_filter}
        {book_status_filter}
    GROUP BY
        dt."date",
        b.created_date::date,
        b.check_in, r."name",
        b.rate_plan_code, b.booking_channel_name, 
        b.check_out, b.booking_reference,
        br.total_revenue, b.total_revenue,
        b.nights,
        b.room_number,
        h."name",
        r."name",r.code,
        b.cancel_date,
        b.booking_status
    ORDER BY
        b.created_date::date,
        dt."date" ASC;
    """
    
    return pd.read_sql_query(detail_query, engine)

# Define the bar chart for booking channels
def create_bar_chart(df):
    df['month'] = pd.to_datetime(df['stay_date']).dt.to_period('M').astype(str)
    fig = go.Figure()
    
    # Group by booking channel and date
    for channel in df['booking_channel_name'].unique():
        channel_df = df[df['booking_channel_name'] == channel]
        monthly_data = channel_df.groupby('month')['number_of_bookings'].sum().reset_index()
        fig.add_trace(go.Bar(
            x=monthly_data['month'],
            y=monthly_data['number_of_bookings'],
            name=channel
        ))
    
    fig.update_layout(
        title='Bookings by Channel Over Time',
        xaxis_title='Month',
        yaxis_title='Number of Bookings',
        barmode='stack',  # Stack bars on top of each other
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1500,  # Set chart width
        height=500  # Set chart height
    )
    
    return fig

# Layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Hotel Booking Dashboard"), width={"size": 6, "offset": 4})
    ]),
    dbc.Row([
        dbc.Col([
            html.Div("Stay Date:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '20px', 'fontFamily': 'Arial'}),
            dcc.DatePickerRange(
                id='stay-date-picker',
                start_date='2024-01-01',
                end_date='2024-07-31',
                display_format='YYYY-MM-DD',  # Format for displaying date
                style={'width': '100%', 'padding': '10px'}
            ),
        ], width=2),

        dbc.Col([
            html.Div("Booking Date:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '20px', 'fontFamily': 'Arial'}),
            dcc.DatePickerRange(
                id='created-date-picker',
                start_date='2024-01-01',
                end_date='2024-07-31',
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
                    'fontSize': '20px',  # Font size
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
                    'fontSize': '20px',  # Font size
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
                    'fontSize': '20px',  # Font size
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
                value=[],  # Default to an empty list (no channels selected)
                multi=True,
                style={
                    'width': '100%', 
                    'fontSize': '20px',  # Font size
                    'fontFamily': 'Arial',  # Font family
                    'color': 'black'  # Font color
                },
                placeholder='Select Booking Status'  # Placeholder text
            ),
        ], width=2),
    ], style={'marginBottom': '20px'}),

    dbc.Row([
        dbc.Col(dcc.Graph(id='booking_heatmap'), width=6),
        dbc.Col(dcc.Graph(id='revenue_heatmap'), width=6)
    ]),
            
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
                        {'name': 'Room Name', 'id': 'room_name'},
                        {'name': 'Room Code', 'id': 'code'},
                        {'name': 'Lead In', 'id': 'date_difference'},
                        {'name': 'Cancel Date', 'id': 'cancel_date'},
                        {'name': 'Stay Date', 'id': 'stay_date'},
                        {'name': 'Booking Date', 'id': 'created_date'},
                        {'name': 'Total Revenue', 'id': 'total_revenue'},
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

    html.Hr(),

    dcc.Loading(
        id="loading-bar-chart",
        type="circle",
        children=[
            dcc.Graph(id='bar-chart')
        ]
    ),

    html.Div(id='hover-data', style={'display': 'none'})  # Placeholder for hover data

], fluid=True)
# Updated callback function to handle room_name filter and preserve layout changes
from datetime import datetime

@app.callback(
    [Output('booking_heatmap', 'figure'),
     Output('revenue_heatmap', 'figure'),
     Output('booking-details', 'data'),
     Output('bar-chart', 'figure'),
     Output('channel-dropdown', 'options'),
     Output('additional-details', 'data'),
     Output('room-dropdown', 'options'),
     Output('rate-dropdown', 'options'),
     Output('book-dropdown', 'options')],
    [Input('hotel-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('room-dropdown', 'value'),
     Input('booking-details', 'active_cell'),
     Input('booking-details', 'data'),
     Input('rate-dropdown', 'value'),
     Input('book-dropdown', 'value'),
     Input('booking_heatmap', 'clickData'),
     Input('revenue_heatmap', 'clickData'),
     Input('booking_heatmap', 'relayoutData'),
     Input('revenue_heatmap', 'relayoutData'),
     Input('stay-date-picker', 'start_date'),
     Input('stay-date-picker', 'end_date'),
     Input('created-date-picker', 'start_date'),
     Input('created-date-picker', 'end_date')]
)
def update_output(selected_hotel, selected_channels, selected_rooms, active_cell, table_data, selected_rate_plan, selected_booking_status, booking_click_data, revenue_click_data, booking_relayout, revenue_relayout, stay_date_start, stay_date_end, created_date_start, created_date_end):
    # Default values
    booking_heatmap = go.Figure()
    revenue_heatmap = go.Figure()
    booking_details_data = []
    bar_chart_fig = go.Figure()
    channel_options = []
    additional_data = []
    room_options = []
    rate_options = []
    book_options = []

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
    df['created_date'] = pd.to_datetime(df['created_date']).dt.date
    df['stay_date'] = pd.to_datetime(df['stay_date']).dt.date

    # Filter data based on selected hotel
    filtered_df = df[df['hotel_id'] == selected_hotel] if selected_hotel else df.copy()

    # Apply date filters
    if stay_date_start and stay_date_end:
        filtered_df = filtered_df[(filtered_df['stay_date'] >= stay_date_start) & (filtered_df['stay_date'] <= stay_date_end)]
    if created_date_start and created_date_end:
        filtered_df = filtered_df[(filtered_df['created_date'] >= created_date_start) & (filtered_df['created_date'] <= created_date_end)]

    # Update dropdown options
    if 'room_name' in filtered_df.columns:
        rooms = filtered_df['room_name'].dropna().unique()
        room_options = [{'label': room, 'value': room} for room in rooms]

    if selected_rooms:
        filtered_df = filtered_df[filtered_df['room_name'].isin(selected_rooms)]

    if 'rate_plan_code' in filtered_df.columns:
        rate_code = filtered_df['rate_plan_code'].dropna().unique()
        sorted_rate_code = sorted([rate for rate in rate_code if rate])
        rate_options = [{'label': rate, 'value': rate} for rate in sorted_rate_code]

    if selected_rate_plan:
        filtered_df = filtered_df[filtered_df['rate_plan_code'].isin(selected_rate_plan)]

    if 'booking_channel_name' in filtered_df.columns and not filtered_df['booking_channel_name'].isnull().all():
        channels = filtered_df['booking_channel_name'].dropna().unique()
        channel_options = [{'label': channel, 'value': channel} for channel in channels]

    if selected_channels:
        filtered_df = filtered_df[filtered_df['booking_channel_name'].isin(selected_channels)]

    if 'booking_status' in filtered_df.columns and not filtered_df['booking_status'].isnull().all():
        bookingstatus = filtered_df['booking_status'].dropna().unique()
        book_options = [{'label': book, 'value': book} for book in bookingstatus]

    if selected_booking_status:
        filtered_df = filtered_df[filtered_df['booking_status'].isin(selected_booking_status)]

    # Create heatmap figures
    booking_heatmap, revenue_heatmap = create_heatmaps(
        df=filtered_df, 
        booking_title='Hotel Booking Heatmap', 
        revenue_title='Hotel Revenue Heatmap', 
        colorscale=custom_colorscale
    )
    
    # Synchronize heatmap zoom and pan
    x_range = y_range = None

    if booking_relayout and 'xaxis.range[0]' in booking_relayout and 'xaxis.range[1]' in booking_relayout:
        x_range = [booking_relayout['xaxis.range[0]'], booking_relayout['xaxis.range[1]']]
        y_range = [booking_relayout['yaxis.range[0]'], booking_relayout['yaxis.range[1]']]
    elif revenue_relayout and 'xaxis.range[0]' in revenue_relayout and 'xaxis.range[1]' in revenue_relayout:
        x_range = [revenue_relayout['xaxis.range[0]'], revenue_relayout['xaxis.range[1]']]
        y_range = [revenue_relayout['yaxis.range[0]'], revenue_relayout['yaxis.range[1]']]

    if x_range and y_range:
        booking_heatmap.update_xaxes(range=x_range)
        booking_heatmap.update_yaxes(range=y_range)
        revenue_heatmap.update_xaxes(range=x_range)
        revenue_heatmap.update_yaxes(range=y_range)

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

    if marker_data:
        stay_date = marker_data['stay_date']
        created_date = marker_data['created_date']

        # Update markers on booking heatmap
        booking_heatmap.add_trace(go.Scatter(
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
        revenue_heatmap.add_trace(go.Scatter(
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

        # Fetch booking details
        booking_details_df = fetch_booking_details(
            stay_date, created_date, selected_hotel, 
            selected_channels if selected_channels else [], 
            selected_rooms if selected_rooms else [], 
            selected_rate_plan if selected_rate_plan else [], 
            selected_booking_status if selected_booking_status else []
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
    
    # Create bar chart
    bar_chart_fig = create_bar_chart(filtered_df)

    # Return updated components
    return booking_heatmap, revenue_heatmap, booking_details_data, bar_chart_fig, channel_options, additional_data, room_options, rate_options, book_options


if __name__ == '__main__':
        serve(server, host='0.0.0.0', port=8000)
