import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI (same as your experiment logs)
mlflow.set_tracking_uri("file:///E:/_DataScienc_KNaik/NLP/Spam_HAM/mlruns")
client = MlflowClient()

experiment_ids = ["291289402592880719", "572695822573749226", "121190556667984861"]
experiments = [client.get_experiment(eid) for eid in experiment_ids]
experiment_options = [{"label": exp.name, "value": exp.experiment_id} for exp in experiments]



# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "MLflow Experiment Dashboard"

app.layout = html.Div([
    html.H1("MLflow Experiments Dashboard"),

    html.Label("Select Experiment:"),
    dcc.Dropdown(id="experiment-dropdown", options=experiment_options, value=experiment_options[0]['value']),
    
    html.Div(id="summary-info", style={"marginTop": 20, "fontWeight": "bold"}),

    html.Br(),
    html.H2("Runs Overview"),

    dcc.Graph(id="accuracy-vs-f1"),
    dcc.Graph(id="precision-vs-recall"),
])

# Callbacks to update graphs based on selected experiment
@app.callback(
    [Output("accuracy-vs-f1", "figure"),
     Output("precision-vs-recall", "figure"),
     Output("summary-info", "children")],
    [Input("experiment-dropdown", "value")]
)
def update_graphs(experiment_id):
    runs = client.search_runs(experiment_ids=[experiment_id])
    if not runs:
        return {}, {}, "No runs found."

    df = pd.DataFrame([
        {
            "run_id": run.info.run_id,
            **run.data.metrics,
            **run.data.params
        }
        for run in runs
    ])
    df.fillna(0, inplace=True)

    fig1 = px.scatter(df, x="Test Accuracy", y="F1-Score_SPAM", text="Model Name",
                      title="Test Accuracy vs F1-Score (SPAM)", height=400,
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Plotly)
    fig2 = px.scatter(df, x="Precision_SPAM", y="Recall_SPAM", text="Model Name",
                      title="Precision vs Recall (SPAM)", height=400,
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Plotly)

    best_acc = df["Test Accuracy"].max()
    run_count = len(df)
    summary = f"Total Runs: {run_count} | Best Test Accuracy: {best_acc:.3f}"

    return fig1, fig2, summary


if __name__ == "__main__":
    app.run(debug=True)

