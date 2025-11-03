#utils/plotting.py

'''Plot on the landscape using Plotly'''

import plotly.graph_objects as go

#############################################################
def make_plot(x_vals, y_vals, Z, x_label="X", y_label="Y", z_label="Confidence"):
    fig = go.Figure(
    data = [
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title='Model Confidence',
                #side='right',
                x=-0.15, 
                xpad=10,
                len=0.75
            )
        )
    ]

    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
    )

    return fig