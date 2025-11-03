#utils/plotting.py

'''Plot on the landscape using Plotly'''

import plotly.graph_objects as go
import numpy as np

#############################################################
def make_plot(x_vals, y_vals, Z, x_label="X", y_label="Y", z_label="Confidence"):
    #Normalize Z shape/type
    x_len, y_len = len(x_vals), len(y_vals)
    Z = np.asarray(Z, dtype=float)

    if Z.ndim !=2:
        raise ValueError(f'Z must be 2D, got shape {Z.shape}')
    
    if Z.shape == (y_len, x_len):
        Zp = Z
    elif Z.shape == (x_len, y_len):
        Zp = Z.T
    else:
        raise ValueError(
            f"Z shape {Z.shape} is incompatible with x({x_len})/y({y_len}). "
            "Check how the surface is computed."
        )

    fig = go.Figure(
    data = [
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=Zp,
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