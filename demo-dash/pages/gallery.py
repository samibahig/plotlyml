import colorsys

from dash import dcc, register_page
import dash_mantine_components as dmc


register_page(
    __name__,
    path="/",
)

layout = dmc.Box(
    [
        dcc.Markdown(
            """        
        # Visual Primitives for ML Monitoring
        Three high-level Plotly functions designed in the spirit of [plotly.express](https://plotly.com/python/plotly-express/),
         filling the gaps in ML observability workflows.        
        """
        ),
        dmc.Divider(m="lg"),
        dmc.SimpleGrid(
            [
                dmc.Card(
                    dcc.Markdown(
                        """
           ###  Distribution Drift
           Compare training vs. live distributions. KL divergence score surfaces feature drift before it degrades model performance.

            ```python
            from plotlyml import distribution_drift
            fig = distribution_drift(ref, cur)
            fig.show()
            ``` 
            """,
                    )
                ),
                dmc.Card(
                    dcc.Markdown(
                        """
            ### Model Disagreement
            Map ensemble variance per sample in a 2-D feature space. High-disagreement regions highlight uncertainty hot-spots for active learning or auditing.
            
            ```python
            from plotlyml import model_disagreement
            fig = model_disagreement(x, y, preds)
            fig.show()
            ```
            """
                    )
                ),
                dmc.Card(
                    dcc.Markdown(
                        """
            ### Quantile Evolution
            Track P10/P50/P90 bands over time. Catches regime changes and volatility spikes that the mean alone would miss.

            ```python
            from plotlyml import quantile_evolution
            fig = quantile_evolution(dates, p50=med)
            fig.show()
           
            ```
            """
                    )
                ),
            ],
            cols={"base": 1, "xl": 3},
        ),
    ]
)
