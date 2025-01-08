from dash import Dash, html
from waitress import serve

# Initialize the Dash app
app = Dash(__name__)

# Layout for the app
app.layout = html.Div([
    html.H1("Embedded Power BI Dashboard"),
    html.Iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiY2NkZTc0YjctNmM3ZC00NGE1LTgzZmYtZTJlMzg0ZjdlOTQzIiwidCI6ImM1MmRiMzE5LTk5ZDgtNGFjMi1hNTBmLTU4ZjdhMDg3N2M0NyJ9&pageName=ReportSection97554cd968fdad03e640",
        style={"width": "100%", "height": "1000px", "border": "none"}
    )
])

# Expose the server object for Gunicorn
server = app.server

# If running locally, use Waitress to serve the app
if __name__ == "__main__":
    serve(app.server, host="0.0.0.0", port=8000)
