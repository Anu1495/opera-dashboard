import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine

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
    r."name" AS room_name,  -- Include room_name in the query
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

# Create the Dash app
app = dash.Dash(__name__)

# Define the custom color scale
custom_colorscale = [
    [0, 'white'],
    [0.1, 'yellow'],
    [0.4, 'blue'],
    [0.6, 'orange'],
    [0.8, 'red'],
    [1, 'brown']
]

def create_heatmap(df, title, colorscale, zmin=None, zmax=None):
    df['booking_channel_name'].fillna('Unknown', inplace=True)
    
    df_agg = df.groupby(['created_date', 'stay_date', 'booking_channel_name']).agg({
        'number_of_bookings': 'sum',
        'total_revenue': 'sum'
    }).reset_index()
    
    pivot_table = df_agg.pivot_table(index=["created_date"], columns="stay_date", values="number_of_bookings", fill_value=0, aggfunc='sum')
    
    customdata_revenue = df_agg.pivot_table(index=["created_date"], columns="stay_date", values="total_revenue", fill_value=0, aggfunc='sum').values
    
    channel_names = df_agg.groupby(['created_date', 'stay_date'])['booking_channel_name'].apply(lambda x: ', '.join(x.unique())).unstack().values
    
    combined_customdata = np.dstack((channel_names, customdata_revenue))
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        customdata=combined_customdata,
        hovertemplate=(
            'Booking Date: %{y}<br>' +
            'Stay Date: %{x}<br>' +
            'Booking Channels: %{customdata[0]}<br>' +
            'Total Number of Bookings: %{z}<br>' +
            'Total Revenue: %{customdata[1]:.2f}'
        ),
        colorscale=colorscale,
        colorbar=dict(title="Number of Bookings", tickvals=[zmin, 0, zmax], ticktext=[f'{zmin}', '0', f'{zmax}']),
        zmin=zmin,
        zmax=zmax
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'font': {
                'size': 20,  # Title font size
                'color': 'black',  # Title font color
                'family': 'Arial',  # Title font family
                'weight': 'bold'  # Title font weight
            },
            'x': 0.5,  # Title x position (centered)
            'xanchor': 'center'  # Title x anchor
        },
        xaxis_title={
            'text': 'Stay Date',
            'font': {
                'size': 16,  # X-axis title font size
                'color': 'black',  # X-axis title font color
                'family': 'Arial'  # X-axis title font family
            }
        },
        yaxis_title={
            'text': 'Booking Date',
            'font': {
                'size': 16,  # Y-axis title font size
                'color': 'black',  # Y-axis title font color
                'family': 'Arial'  # Y-axis title font family
            }
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,  # Set chart width
        height=800,  # Set chart height
        xaxis=dict(
            scaleanchor='y'
        ),
        yaxis=dict(
            scaleratio=1,
            scaleanchor='x'
        ),
    )
    
    return fig


def fetch_booking_details(stay_date, created_date, selected_hotel, selected_channels, selected_rooms):
    channel_filter = f"AND b.booking_channel_name IN ({', '.join(f'\'{channel}\'' for channel in selected_channels)})" if selected_channels else ""
    room_filter = f"AND r.name IN ({', '.join(f'\'{room}\'' for room in selected_rooms)})" if selected_rooms else ""
    
    detail_query = f"""
    SELECT
        dt."date" AS stay_date,
        b.created_date::date,
        b.booking_reference,
        COALESCE(br.total_revenue, b.total_revenue / (CASE WHEN b.nights=0 THEN 1 ELSE b.nights END)) AS total_revenue,
        COUNT(b.booking_reference) AS number_of_bookings,
        b.booking_status,
        EXTRACT(DAY FROM (dt."date" - b.created_date::date)) AS date_difference,
        b.booking_channel_name,
        r."name" AS room_name,
        b.check_in,
        b.check_out,
        b.rate_plan_code,
        h."name",
        CASE 
            WHEN b.check_out > b.check_in THEN (b.check_out - b.check_in)::integer
            ELSE 0
        END AS NumberOfNights
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
        dt."date" = '{stay_date}'
        AND b.created_date = '{created_date}'
        AND b.hotel_id = {selected_hotel}
        {channel_filter}
        {room_filter}
    GROUP BY
        dt."date",
        b.created_date::date,
        b.check_in, r."name",
        b.rate_plan_code, b.booking_channel_name, 
        b.check_out, b.booking_reference,
        br.total_revenue, b.total_revenue,
        b.nights,
        h."name",
        r."name",
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
# Layout of the Dash app
app.layout = html.Div([
    html.H1(
        'Opera Dashboard',
        style={
            'fontSize': '20px',  # Font size
            'fontFamily': 'Arial',
            'font-weight': 'bold',
            'textAlign': 'center',    # Font family
            'color': 'black', # Optional: Center align the title
            'marginBottom': '20px' # Optional: Add margin below the title
        }
    ),

    html.Div([
        html.Div([
            html.Div("Stay Date:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.DatePickerRange(
                id='stay-date-picker',
                start_date='2024-01-01',
                end_date='2024-07-31',
                display_format='YYYY-MM-DD',  # Format for displaying date
                style={'width': '100%', 'padding': '10px'}
            ),
            html.Div("Booking Date:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.DatePickerRange(
                id='created-date-picker',
                start_date='2024-01-01',
                end_date='2024-07-31',
                display_format='YYYY-MM-DD',  # Format for displaying date
                style={'width': '100%', 'padding': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

       html.Div([
    dcc.Dropdown(
        id='hotel-dropdown',
        options=hotel_options,
        value=hotel_options[0]['value'],  # Default value
        style={
            'width': '100%', 
            'marginBottom': '10px', 
            'fontSize': '20px',  # Font size
            'fontFamily': 'Arial',  # Font family
            'color': 'black'  # Font color
        },
        clearable=False
    ),
    dcc.Dropdown(
        id='channel-dropdown',
        options=[],  # Initially empty
        value=[],  # Default to an empty list (no channels selected)
        multi=True,  # Enable multi-select
        style={
            'width': '100%', 
            'marginBottom': '10px', 
            'fontSize': '20px',  # Font size
            'fontFamily': 'Arial',  # Font family
            'color': 'black'  # Font color
        }
    ),
    dcc.Dropdown(
        id='room-dropdown',
        multi=True,
        style={
            'width': '100%', 
            'fontSize': '20px',  # Font size
            'fontFamily': 'Arial',  # Font family
            'color': 'black'  # Font color
        }
    )
], style={'width': '65%', 'display': 'inline-block', 'padding': '10px'})
,

    ], style={'padding': '20px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='heatmap'),
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.H2('Booking Details'),
            dash_table.DataTable(
                id='booking-details',
                columns=[
                    {'name': 'Booking Reference', 'id': 'booking_reference'},
                    {'name': 'Booking Status', 'id': 'booking_status'},
                    {'name': 'Room Name', 'id': 'room_name'},
                    {'name': 'Lead In', 'id': 'date_difference'},
                    {'name': 'Stay Date', 'id': 'stay_date'},
                    {'name': 'Booking Date', 'id': 'created_date'},
                    {'name': 'Hotel Name', 'id': 'name'},
                    {'name': 'Booking channel', 'id': 'booking_channel_name'},
                    {'name': 'Check-in Date', 'id': 'check_in'},
                    {'name': 'Check-out Date', 'id': 'check_out'},
                    {'name': 'Rate Plan', 'id': 'rate_plan_code'},
                    {'name': 'Number of Nights', 'id': 'number_of_bookings'},
                    {'name': 'Total Revenue', 'id': 'total_revenue'},
                ],
                style_table={'overflowX': 'auto', 'fontSize': 14},
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '18px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold', 'fontSize': 18},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)',
                    },
                ],
                style_as_list_view=True
            )
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'flex-direction': 'row-reverse'}),

    html.Hr(),

    dcc.Graph(id='bar-chart'),

    html.Div(id='hover-data', style={'display': 'none'})  # Placeholder for hover data
    
])

# Updated callback function to handle room_name filter and preserve layout changes
from datetime import datetime

@app.callback(
    [Output('heatmap', 'figure'),
     Output('booking-details', 'data'),
     Output('bar-chart', 'figure'),
     Output('channel-dropdown', 'options'),
     Output('room-dropdown', 'options')],
    [Input('hotel-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('room-dropdown', 'value'),
     Input('heatmap', 'clickData'),
     Input('heatmap', 'relayoutData'),
     Input('stay-date-picker', 'start_date'),
     Input('stay-date-picker', 'end_date'),
     Input('created-date-picker', 'start_date'),
     Input('created-date-picker', 'end_date')]
)
def update_output(selected_hotel, selected_channels, selected_rooms, click_data, relayout_data, stay_date_start, stay_date_end, created_date_start, created_date_end):
    # Convert date strings to datetime.date objects
    stay_date_start = datetime.strptime(stay_date_start, '%Y-%m-%d').date() if stay_date_start else None
    stay_date_end = datetime.strptime(stay_date_end, '%Y-%m-%d').date() if stay_date_end else None
    created_date_start = datetime.strptime(created_date_start, '%Y-%m-%d').date() if created_date_start else None
    created_date_end = datetime.strptime(created_date_end, '%Y-%m-%d').date() if created_date_end else None

    # Ensure dataframe date columns are in datetime format
    df['created_date'] = pd.to_datetime(df['created_date']).dt.date
    df['stay_date'] = pd.to_datetime(df['stay_date']).dt.date

    # Filter data based on selected hotel
    if selected_hotel:
        filtered_df = df[df['hotel_id'] == selected_hotel]
    else:
        filtered_df = df

    # Apply date filters only if dates are provided
    if stay_date_start and stay_date_end:
        filtered_df = filtered_df[(filtered_df['stay_date'] >= stay_date_start) & (filtered_df['stay_date'] <= stay_date_end)]
    if created_date_start and created_date_end:
        filtered_df = filtered_df[(filtered_df['created_date'] >= created_date_start) & (filtered_df['created_date'] <= created_date_end)]

    # Update channel dropdown options
    if 'booking_channel_name' in filtered_df.columns and not filtered_df['booking_channel_name'].isnull().all():
        channels = filtered_df['booking_channel_name'].dropna().unique()
        channel_options = [{'label': channel, 'value': channel} for channel in channels]
    else:
        channel_options = []

    # Update room dropdown options
    if 'room_name' in filtered_df.columns:
        rooms = filtered_df['room_name'].dropna().unique()
        room_options = [{'label': room, 'value': room} for room in rooms]
    else:
        room_options = []

    # Apply filters for selected channels and rooms
    if selected_channels:
        filtered_df = filtered_df[filtered_df['booking_channel_name'].isin(selected_channels)]
    if selected_rooms:
        filtered_df = filtered_df[filtered_df['room_name'].isin(selected_rooms)]

    # Create heatmap figure
    heatmap_fig = create_heatmap(filtered_df, title='Hotel Booking Heatmap', colorscale=custom_colorscale)
    
    # Check if there is relayoutData (zoom or pan) and preserve it
    if relayout_data:
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            heatmap_fig.update_xaxes(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']])
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            heatmap_fig.update_yaxes(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])

    # Fetch booking details if heatmap is clicked
    if click_data:
        stay_date = click_data['points'][0]['x']
        created_date = click_data['points'][0]['y']
        booking_details_df = fetch_booking_details(stay_date, created_date, selected_hotel, selected_channels if selected_channels else [], selected_rooms if selected_rooms else [])
        booking_details_data = booking_details_df.to_dict('records')
    else:
        booking_details_data = []

    # Create bar chart
    bar_chart_fig = create_bar_chart(filtered_df)

    return heatmap_fig, booking_details_data, bar_chart_fig, channel_options, room_options


if __name__ == '__main__':
    app.run_server(debug=True)
