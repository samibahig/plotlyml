from dash import Dash, Input, Output, State, callback, page_container
import dash_mantine_components as dmc
from dash_iconify import DashIconify

app = Dash(
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/gh/snehilvj/dash-mantine-components@master/src/ts/components/extensions/codehighlight/fragments/dmc-code.css"
    ],
)

theme_toggle = dmc.ColorSchemeToggle(
    lightIcon=DashIconify(icon="radix-icons:sun", width=20),
    darkIcon=DashIconify(icon="radix-icons:moon", width=20),
    color="yellow",
    size="lg",
)

header = dmc.AppShellHeader(
    dmc.Group(
        [
            dmc.Group(
                [
                    dmc.Burger(
                        id="burger",
                        hiddenFrom="sm",
                    ),
                    dmc.Title("plotlyml", order=2),
                    dmc.Text(
                        "ML Monitoring Visualization Primitives",
                        visibleFrom="md",
                    ),
                ],
                gap="md",
            ),
            dmc.Group(
                [
                    dmc.Anchor(
                        dmc.ActionIcon(DashIconify(icon="mdi:github", width=18)),
                        href="https://github.com/samibahig/plotlyml",
                        target="_blank",
                        variant="subtle",
                    ),
                    theme_toggle,
                ],
                gap="sm",
            ),
        ],
        justify="space-between",
        h="100%",
        px="xl",
    )
)


app.layout = dmc.MantineProvider(
    dmc.AppShell(
        [
            dmc.AppShellHeader(header),
            dmc.AppShellNavbar(
                id="navbar",
                p="md",
                children=[
                    dmc.NavLink(label="Gallery", href="/", active="partial"),
                    dmc.NavLink(
                        label="Distribution Drift",
                        href="/distribution-drift",
                        active="partial",
                    ),
                    dmc.NavLink(
                        label="Model Disagreement",
                        href="/model-disagreement",
                        active="partial",
                    ),
                    dmc.NavLink(
                        label="Quantile Evolution",
                        href="/quantile-evolution",
                        active="partial",
                    ),
                ],
            ),
            dmc.AppShellMain(page_container),
        ],
        id="appshell",
        header={"height": 80},
        navbar={
            "width": 200,
            "breakpoint": "sm",
            "collapsed": {"mobile": True},
        },
        padding="md",
    ),
    theme={"components": {"Card": {"defaultProps": {"withBorder": True, "mt": "xl"}}}},
)


@callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def toggle_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


if __name__ == "__main__":
    app.run(debug=True)
