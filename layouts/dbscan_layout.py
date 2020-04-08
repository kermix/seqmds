import dash_core_components as dcc
import dash_html_components as html

def ganerate_layout():
    layout = html.Div([
        html.Table([
            html.Tbody([html.Tr([
                html.Td([html.P("Type a number of componenets/dimmensions for MDS. Only positive integers.")]),
                html.Td([html.P("Type a eps distance for creating clusters in DBSCAN procedure.")]),
                html.Td([html.P("Type a minimum size of cluster in DBSCAN procedure. Only positive integers.")]),
                html.Td()
            ]),
                html.Tr([
                    html.Td([
                        dcc.Input(
                            id='no-components',
                            placeholder='Number of components/dimmensions',
                            type='number',
                            value='2',
                            min=2,
                            max=3,
                            step=1
                        )
                    ]),
                    html.Td([
                        dcc.Input(
                            id='db-scan-eps',
                            placeholder='dbscan eps distance',
                            type='number',
                            value='0.15',
                            min=0.01,
                            step=.01
                        )
                    ]),
                    html.Td([
                        dcc.Input(
                            id='db-scan-min-cluster-size',
                            placeholder='dbscan min size of cluster',
                            type='number',
                            value='2',
                            min=1,
                            step=1
                        ),
                    ]),
                    html.Td([
                        dcc.RadioItems(
                            options=[
                                {'label': 'Plot Clusters', 'value': 'T'},
                                {'label': 'Don\'t plot clusters', 'value': 'F'},
                            ],
                            value='F',
                            id='plot-clusters-checkbox'
                        )
                    ])
                ])])
        ], style={'width': '100%'}),
        dcc.Loading(id='loading-1',
                    children=[html.Div(id='mds-signal', style={'display': 'none'}),
                              html.Div(id='clustering-signal', style={'display': 'none'}),
                              dcc.Graph(id='indicator-graphic',
                                        # style={'height': '85vh'}
                                        ),
                              dcc.Graph(id='reachability-graphic')
                              ])
    ])
    return layout
