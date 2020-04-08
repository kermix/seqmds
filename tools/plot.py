import itertools
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gen_cluster_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgb({r}, {g}, {b})"

def plot3d(data, clusters, labels):
    # TODO: multiple traces for different colors
    plot_data = go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        text=labels,
        mode='markers',
        marker=dict(
            size=5,
            color=clusters,  # set color to an array/list of desired values
            opacity=0.8,
            colorscale="Viridis"
        )
    )


    return plot_data


def plot2d(data, labels, name, showlegend=True, color='rgb(0,0,0)'):
    plot_data = go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        text=labels,
        mode='markers',
        marker=dict(
            size=7,
            color=color,  # set color to an array/list of desired values
            opacity=0.8,
        ),
        legendgroup='a',
        showlegend=showlegend,
        name=name,
    )

    return plot_data


def plot2d_subplots(data, clusters, labels, title=""):
    dims_combination = list(itertools.combinations(range(data.shape[1]), 2))
    n_combinations = len(dims_combination)
    n_rows = 1 if n_combinations == 1 else n_combinations // 3 + (n_combinations % 3 > 0)
    n_cols = 1 if n_combinations == 1 else 3

    fig = make_subplots(rows=n_rows, cols=n_cols)
    colors = ["rgb(0,0,0)"]+[gen_cluster_color() for c in clusters[1:]]
    for i, c in enumerate(dims_combination):
        row = (i // 3) + 1
        col = (i % 3) + 1
        for cluster in range(min(clusters), max(clusters) + 1):


            mask = clusters == cluster
            color=colors[cluster+1]
            fig.add_trace(plot2d(
                np.column_stack(
                    (
                        data[mask, c[0]],
                        data[mask, c[1]]
                    )
                ),
                labels[mask],
                " ".join(("cluster", str(cluster))) if cluster != -1 else "Not clustered",
                False if row != 1 or col != 1 else True,
                color
            ),
                row=row,
                col=col
            )
        fig.update_xaxes(title_text="PC" + str(c[0] + 1), row=row, col=col)
        fig.update_yaxes(title_text="PC" + str(c[1] + 1), row=row, col=col)

    xaxes = [(k, fig['layout'][k]['title']) for k in fig['layout'] if 'xaxis' == k[:5]]
    yaxes = [(k, fig['layout'][k]['title']) for k in fig['layout'] if 'yaxis' == k[:5]]

    fig.layout = gen_proper_subplot_layout(xaxes, yaxes)
    return fig

def gen_proper_subplot_layout(xaxes, yaxes):
    new_xaxes, new_yaxes = dict(), dict()
    col_width = .3
    row_height = 1.0 / ((len(yaxes) // 3)+1)
    spacer = .04
    for i, axis in enumerate(xaxes):
        ax, title = axis
        no_column = i % 3
        d_start = (col_width+spacer)*no_column
        d_end = d_start+col_width
        new_xaxes[ax] = dict(
            domain=[d_start, d_end],
            anchor=('y' + str(ax[5:])),
            title=title
        )

    for j, axis in enumerate(yaxes):
        ay, title = axis
        # no_row = j // 3
        # d_start = (row_height)*no_row
        # d_end = d_start+row_height
        new_yaxes[ay] = dict(
            # domain=[d_start, d_end],
            scaleanchor=('x' + str(j+1)),
            anchor=('x' + str(j+1)),
            title=title
        )

    new_axes = {**new_xaxes, **new_yaxes}

    return go.Layout(new_axes)

def plot_dbscan_clusters(fig, eps):
    new_fig = go.Figure(fig)
    plot_data = fig["data"]
    eps = float(eps)
    for trace in filter(lambda t: t['name'] != 'Not clustered', plot_data):
        points = zip(trace['x'], trace['y'])
        color = trace['marker']['color']
        xaxis = trace['xaxis'][1:]
        i = (int(xaxis) if xaxis else 1) - 1
        row = (i // 3) + 1
        col = (i % 3) + 1
        for p_x, p_y in points:
            new_fig.add_shape(
                    go.layout.Shape(
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=p_x+eps,
                        y0=p_y-eps,
                        x1=p_x-eps,
                        y1=p_y+eps,
                        fillcolor=color,
                        opacity=0.05,
                        layer="below",
                        line_width=0
                    ), row=row, col=col)
    return new_fig
