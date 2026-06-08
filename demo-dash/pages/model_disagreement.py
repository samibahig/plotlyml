from dash import callback, Input, Output, dcc, register_page
import dash_mantine_components as dmc

from plotlyml import model_disagreement
from .utils import generate_disagreement_data

register_page(
    __name__,
    path="/model-disagreement",
)

controls = dmc.SimpleGrid(
    [
        dmc.InputWrapper(
            dmc.Slider(
                id="threshold",
                min=0.005,
                max=0.2,
                value=0.05,
                step=0.005,
            ),
            label="Alert Threshold (Variance)",
        ),
        dmc.InputWrapper(
            dmc.Slider(
                id="samples",
                min=100,
                max=500,
                value=300,
                step=50,
            ),
            label="Number of Samples",
        ),
    ],
    cols=2,
)

api = """
from plotlyml import model_disagreement

fig = model_disagreement(
    x,                    # 1-D array: reduced feature dim 1
    y,                    # 1-D array: reduced feature dim 2
    predictions,          # 2-D array: shape (n_samples, n_models)
    sample_ids=None,      # optional list of string IDs
    true_labels=None,     # optional ground-truth labels
    threshold=0.05,       # variance alert threshold
    x_label="Dim 1",
    y_label="Dim 2",
    colorscale="Viridis",
    template="plotly_dark",
)
fig.show()
"""

layout = dmc.Stack(
    [
        dmc.Title("Model Disagreement Map", order=1),
        dmc.Text(
            "Visualize per-sample ensemble variance in a 2-D feature space. High variance indicates high uncertainty."
        ),
        dmc.Card(
            [
                dmc.Title("Simulation Controls", order=4, mb="lg"),
                controls,
            ]
        ),
        dmc.Card(dcc.Graph(id="disagreement-graph")),
        dmc.Card(
            [
                dmc.Title("Component API", order=4),
                dmc.CodeHighlight(
                    language="python",
                    code=api,
                    withCopyButton=True,
                ),
            ]
        ),
        dmc.Card(
            dcc.Markdown(
                """
                #### Use Cases
                - **Active learning** — prioritize labeling samples where the ensemble disagrees most
                - **Uncertainty-aware deployment** — flag high-variance samples for human review at inference time
                - **Model debugging** — identify clusters where the ensemble is systematically confused
                - **Distribution shift detection** — high-disagreement regions often correlate with out-of-distribution inputs
                """
            ),
        ),
    ]
)


@callback(
    Output("disagreement-graph", "figure"),
    Input("threshold", "value"),
    Input("samples", "value"),
    Input("theme-toggle", "computedColorScheme"),
)
def update_graph(threshold, samples, theme):

    xs, ys, preds_list = generate_disagreement_data(samples)
    template = "plotly_dark" if theme == "dark" else "plotly_white"

    return model_disagreement(
        xs,
        ys,
        preds_list,
        threshold=threshold,
        x_label="UMAP Dim 1",
        y_label="UMAP Dim 2",
        title="Ensemble Disagreement Map",
        template=template
    )
