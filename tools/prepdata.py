import pandas as pd
import numpy as np
import itertools

aminoacids = list('ACDEFGHIKLMNPQRSTVWYX-')
amino_bin = np.zeros((len(aminoacids), len(aminoacids)))
np.fill_diagonal(amino_bin, 1)
amino_bin = {aminoacids[i]: arr for i, arr in enumerate(amino_bin)}


def save_important_seqs(in_dir, out_dir, n_components, lim):
    mat = pd.read_csv(in_dir, index_col=0, header=0, nrows=n_components)
    mat = abs(mat)
    lim = abs(lim)
    is_important = mat.T[(mat > lim).any()].index  # to moze byc nienajlepsze, bo 0.7 w PC7 =/= 0.7 w PC1
    np.savetxt(out_dir, is_important.to_numpy(dtype=np.str), fmt="%s")
    return is_important.to_numpy()


def gen_cols(seq_l):
    products = itertools.product([str(x) for x in range(0, seq_l)], aminoacids)
    return list([''.join(p) for p in products])


def data_to_bool_vec(data):
    columns = gen_cols(len(data.iloc[0].values[0]))
    new_data = []
    for i_seq, seq in data.iterrows():
        seq = seq.values[0]
        new_data.append(np.concatenate(list(map(lambda a: amino_bin[a], seq))))
    return pd.DataFrame(new_data, index=data.index, columns=columns, dtype='int64')


def center_bool_vec_data(data):
    # for test, slow, returning np.ndarray
    c = data.values - data.values.mean(axis=0)
    m = np.zeros((c.shape[0], c.shape[1], len(c[0][0])))
    for i, row in enumerate(c):
        for j, column in enumerate(row):
            for k, elem in enumerate(column):
                m[i, j, k] = elem
    m = m.reshape(m.shape[0], m.shape[1] * m.shape[2])  # może spróbować by wyliczania wartości na podstawie wlasności
    return m


def center_bool_vec(data):
    return data - data.mean()


def read_seq_data(in_dir):
    seqs = {}
    with open(in_dir, 'r') as file:
        seq_name = None
        for line in file.readlines():
            line = line.strip()
            if line.startswith('>'):
                seq_name = line[1:]
                seqs[seq_name] = ''
            else:
                seqs[seq_name] += line
    return pd.DataFrame(seqs, index=[0]).T
