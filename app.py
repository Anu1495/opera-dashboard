from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Embedded Power BI Dashboard"),
    html.Iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiY2NkZTc0YjctNmM3ZC00NGE1LTgzZmYtZTJlMzg0ZjdlOTQzIiwidCI6ImM1MmRiMzE5LTk5ZDgtNGFjMi1hNTBmLTU4ZjdhMDg3N2M0NyJ9&pageName=ReportSection97554cd968fdad03e640",
        style={"width": "100%", "height": "1000px", "border": "none"}
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
