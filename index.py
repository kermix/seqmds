import os
import sys
import argparse

import pandas as pd
import h5py

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import dbscan_layout
import layouts.dbscan_callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return dbscan_layout.ganerate_layout()
    else:
        return '404'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in', help='Input file directory', required=True)
    ap.add_argument('-d', '--data_file', help='Directory of h5 data file to read/create.', default=None)

    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(1)

    args = vars(ap.parse_args())

    if args['data_file'] is None:
        app.FILE = h5py.File('seqdata_file', driver='core', mode='a', backing_store=False)
    else:
        df = open(args['data_file'], mode='w+b')
        app.FILE = h5py.File(df, mode='a')

    if os.path.exists(args['in']):
        matrix = pd.read_csv(args['in'], sep=",", index_col=0, header=None)
        matrix.columns = matrix.index

        app.matrix = matrix

        app.run_server(debug=True, host='0.0.0.0')
    else:
        print("File {} not found".format(args['in']))