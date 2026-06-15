from dash import callback, Input, Output, dcc, register_page
import dash_mantine_components as dmc
from plotlyml import distribution_drift
from .utils import generate_samples

register_page(
    __name__,
    path="/distribution-drift",
)

controls = dmc.SimpleGrid(
    [
        dmc.InputWrapper(
            dmc.Slider(
                id="mean-shift",
                min=-2,
                max=3,
                value=0.8,
                step=0.05,
            ),
            label="Mean Shift",
        ),
        dmc.InputWrapper(
            dmc.Slider(
                id="std-scale",
                min=0.5,
                max=3,
                value=1.2,
                step=0.05,
            ),
            label="Current Std Scale",
        ),
        dmc.InputWrapper(
            dmc.Slider(
                id="threshold",
                min=0.01,
                max=1,
                value=0.1,
                step=0.01,
            ),
            label="KL Divergence Treshold",
        ),
        dmc.InputWrapper(
            dmc.Slider(
                id="samples",
                min=200,
                max=2000,
                value=800,
            ),
            label="Samples per distribution",
        ),
    ],
    cols=2,
)

api = """
from plotlyml import distribution_drift

fig = distribution_drift(
    reference,               # list | np.ndarray | pd.Series
    current,                 # list | np.ndarray | pd.Series
    bins=50,                 # histogram bins
    divergence_threshold=0.1,# KL alert threshold
    reference_name="Reference",
    current_name="Current",
    x_label="Feature Value",
    title=None,              # auto-generated if omitted
    template="plotly_dark",
)
fig.show()
"""

layout = dmc.Stack(
    [
        dmc.Title("Distribution Drift Plot", order=1),
        dmc.Text(
            "Compare two distributions and surface divergence — training vs. inference, A/B test groups, pre/post deployment."
        ),
        dmc.Card(
            [
                dmc.Title("Simulation Contriols", order=4, mb="lg"),
                controls,
            ]
        ),
        dmc.Card(dcc.Graph(id="drift-graph")),
        dmc.Card(
            [
                dmc.Title(
                    "Component API",
                    order=4,
                ),
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
            - **Feature store monitoring** — detect when a feature's live distribution drifts from the training snapshot
            - **A/B test sanity check** — confirm groups are comparable before reading metrics
            - **Post-deployment audit** — flag model inputs that fall outside the training domain
            - **Data pipeline QA** — catch upstream ETL regressions that shift numerical columns
            """
            ),
        ),
    ]
)


@callback(
    Output("drift-graph", "figure"),
    Input("mean-shift", "value"),
    Input("std-scale", "value"),
    Input("threshold", "value"),
    Input("samples", "value"),
    Input("theme-toggle", "computedColorScheme"),
)
def update_graph(mean_shift, std_scale, threshold, samples, theme):

    ref = generate_samples(0, 1, samples)
    cur = generate_samples(mean_shift, std_scale, samples)
    template = "plotly_dark" if theme == "dark" else "plotly_white"

    return distribution_drift(
        ref,
        cur,
        divergence_threshold=threshold,
        template=template,
    )
