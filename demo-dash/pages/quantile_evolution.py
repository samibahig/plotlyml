from dash import callback, Input, Output, dcc, register_page
import dash_mantine_components as dmc
from plotlyml import quantile_evolution
from .utils import generate_quantile_data

register_page(
    __name__,
    path="/quantile-evolution",
)

controls = dmc.SimpleGrid(
    [
        dmc.Switch(
            id="show-mean",
            label="Overlay Mean Line",
            checked=True,
        ),
        dmc.Switch(
            id="show-inner",
            label="Show P25-P75 Band",
            checked=True,
        ),
        dmc.InputWrapper(
            dmc.Slider(
                id="weeks",
                min=12,
                max=104,
                value=52,
                step=4,
            ),
            label="Number of Weeks",
        ),
    ],
    cols=3,
)

api = """
from plotlyml import quantile_evolution

fig = quantile_evolution(
    timestamps,
    p50=median_values,
    p10=p10_values,
    p90=p90_values,
    p25=p25_values,
    p75=p75_values,
    mean=mean_values,
    show_mean=False,
    volatility_multiplier=1.5,
    template="plotly_dark",
)
fig.show()
"""

layout = dmc.Stack(
    [
        dmc.Title("Quantile Evolution Plot", order=1),
        dmc.Text(
            "Track P10/P50/P90 bands over time and identify regime changes and volatility spikes."
        ),
        dmc.Card(
            [
                dmc.Title("Simulation Controls", order=4, mb="lg"),
                controls,
            ]
        ),
        dmc.Card(dcc.Graph(id="quantile-graph")),
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
                - **Service reliability** — detect when P99 latency spikes while P50 remains stable
                - **ML prediction confidence** — track how certainty evolves after deployment
                - **Clinical trials** — compare endpoint distributions across cohorts over time
                - **Financial risk** — monitor VaR and CVaR evolution under changing market conditions
                - **Data quality** — detect increasing quantile spread caused by upstream instability
                """
            ),
        ),
    ]
)


@callback(
    Output("quantile-graph", "figure"),
    Input("show-mean", "checked"),
    Input("show-inner", "checked"),
    Input("weeks", "value"),
    Input("theme-toggle", "computedColorScheme"),
)
def update_graph(show_mean, show_inner, weeks, theme):
    (
        timestamps,
        p10s,
        p25s,
        p50s,
        p75s,
        p90s,
        means,
    ) = generate_quantile_data(weeks)
    template = "plotly_dark" if theme == "dark" else "plotly_white"

    return quantile_evolution(
        timestamps,
        p50=p50s,
        p10=p10s,
        p90=p90s,
        p25=p25s if show_inner else None,
        p75=p75s if show_inner else None,
        mean=means,
        show_mean=show_mean,
        y_label="Metric Value",
        title="Metric Quantile Evolution",
        template=template
    )
