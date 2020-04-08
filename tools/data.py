import re
import os
import sys

from io import StringIO

import pandas as pd


def read_distance_matrix(in_dir, out_name='saned_distance_matrix.csv'):
    try:
        with open(in_dir, 'r') as file:
            data = _saned_clustalo_dist_file(file)
    except OSError:
        print("Cannot open directory: {}".format(in_dir))
        sys.exit(-1)

    out_dir = os.path.dirname(in_dir)
    out_filepath = os.path.join(out_dir, out_name)

    matrix = _prepare_df(data, out_filepath)
    return matrix


def _saned_clustalo_dist_file(file):
    data = ''
    regex = re.compile(r'(?:(?=[^\r\n])\s+)')
    for line in file.readlines()[1:]:
        data += re.sub(regex, ' ', line)
    return data

def _prepare_df(data, out_dir):
    matrix = pd.read_csv(StringIO(data), sep=" ", index_col=0, header=None)
    matrix.columns = matrix.index.values
    matrix.to_csv(out_dir, header=True)
    return matrix


