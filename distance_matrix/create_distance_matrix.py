from Bio import SeqIO
import itertools
import numpy as np
import ctypes

from tempfile import mkdtemp
import os.path as path

import multiprocessing as mp
from collections import deque
import time

sequences = SeqIO.to_dict(SeqIO.parse("/home/mateusz/pca/clear_in.fa", "fasta"))
indexes = list(sequences.keys())


def optimize_sequence_dict(sequences, indexes):
    seqs = {index: str(sequences[index]) for index in indexes}
    return seqs, list(seqs.keys())


sequences, indexes = optimize_sequence_dict(sequences, indexes[:])


def shared_zeros(n1, n2):
    # create a 2D numpy array which can be then changed in different threads
    shared_array_base = mp.Array(ctypes.c_float, n1 * n2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n1, n2)
    return shared_array


class singleton:
    arr = None


def calc_dist(i12):
   s1 = sequences[indexes[i12[0]]]
   s2 = sequences[indexes[i12[1]]]
   singleton.arr[i12[0]][i12[1]] = 1


def main():
    filename = path.join(mkdtemp(), 'distmatrix.dat')

    pairs = itertools.combinations(range(len(indexes)), 2)

    singleton.arr = shared_zeros(len(indexes), len(indexes))
    n_pairs = (np.math.factorial(len(indexes))) / (np.math.factorial(2) * np.math.factorial(len(indexes) - 2))

    pool = mp.Pool(12)
    deque(pool.imap_unordered(calc_dist, pairs, chunksize=10000), 0)


if __name__=='__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
