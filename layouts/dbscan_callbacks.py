import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.manifold import MDS

from app import app
from tools.clustering import dbscan
from tools.plot import plot2d_subplots, plot_dbscan_clusters, gen_cluster_color

import numpy as np


@app.callback(Output("mds-signal", "children"),
              [Input('no-components', 'value')])
def mds_data(components):
    path = f"/mds/components_{components}_data"
    if path not in app.FILE:
        app.logger.info("mds")
        components = int(components)
        emb = MDS(n_components=components, dissimilarity='precomputed', random_state=1)
        transformed = emb.fit_transform(app.matrix)

        dset = app.FILE.create_dataset(path, data=transformed)

    return path


@app.callback(Output("clustering-signal", "children"),
              [Input('db-scan-eps', 'value'),
               Input('db-scan-min-cluster-size', 'value'),
               Input('mds-signal', 'children')],
              [State('no-components', 'value')])
def cluster_data(eps, min_sample, mds_path, components):
    path = f"/mds/components_{components}/dbscan_{eps}_{min_sample}_data"
    if path not in app.FILE:
        eps = float(eps)
        min_sample = float(min_sample)
        app.logger.info("dbscan")
        result, _ = dbscan(app.FILE[mds_path], eps, min_sample)
        dset = app.FILE.create_dataset(path, data=result, chunks=True)

    return path


@app.callback(
    [Output('indicator-graphic', 'figure'),
     Output('reachability-graphic', 'figure')],
    [Input("clustering-signal", "children"),
     Input('plot-clusters-checkbox', 'value')],
    [State('mds-signal', 'children'),
     State('db-scan-eps', 'value'),
     State('no-components', 'value')])
def update_graph(clustering_path, plot_clusters, mds_path, eps, no_components):
    clust = OPTICS(min_samples=2, xi=.05, metric='euclidean', p=None)

    transformed = app.FILE[mds_path]
    space = np.arange(len(transformed))
    clust.fit(transformed)

    a = cluster_optics_dbscan(reachability=clust.reachability_,
                              core_distances=clust.core_distances_,
                              ordering=clust.ordering_, eps=float(eps))

    # labels = clust.fit_predict(data)
    reachability = clust.core_distances_[clust.ordering_]
    reachability[reachability == np.inf] = 0
    y_pred_dbscan = a[clust.ordering_]

    no_clusters = np.max(y_pred_dbscan)
    colors = [gen_cluster_color() for _ in range(no_clusters + 1)]

    names = app.matrix.index[clust.ordering_]

    fig = go.Figure()

    for klass, color in zip(range(no_clusters + 1), colors):
        Xk = space[y_pred_dbscan == klass]
        Rk = reachability[y_pred_dbscan == klass]
        fig.add_trace(go.Scatter(x=Xk, y=Rk,
                                 mode='markers',
                                 name='markers',
                                 marker=dict(color=color),
                                 text=names[y_pred_dbscan == klass]))
    fig.add_trace(go.Scatter(x=space[y_pred_dbscan == -1], y=reachability[y_pred_dbscan == -1],
                             mode='markers', marker=dict(color='rgb(0,0,0)'),
                             name='-1',
                             text=names[y_pred_dbscan == -1]))

    app.logger.info("ploting")
    a = cluster_optics_dbscan(reachability=clust.reachability_,
                              core_distances=clust.core_distances_,
                              ordering=clust.ordering_, eps=float(eps))

    plots = plot2d_subplots(transformed, a[:], app.matrix.index)
    if plot_clusters == 'T':
        plots = plot_dbscan_clusters(plots, eps)

    return plots, fig
