from mpi4py import MPI
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
# from memory_profiler import profile

from Bio import Align

import numpy as np
import itertools
import h5py
import time
import pandas as pd

start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.rank
n_processes = comm.size

#sequences = SeqIO.to_dict(SeqIO.parse("/home/mateusz/pca/clear_in.fa", "fasta"))
sequences = SeqIO.to_dict(SeqIO.parse("/home/mateusz/pca/test/nowy_2/seq_Mateusz.fa", "fasta"))
indexes = list(sequences.keys())

# @profile
def optimize_sequence_dict(sequences, indexes):
    seqs = {index: str(format(sequences[index].seq)) for index in indexes}
    return seqs, list(seqs.keys())

sequences, indexes = optimize_sequence_dict(sequences, indexes[:])
size = len(indexes)

f = h5py.File("/tmp/aaaa.hd5f", 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (size, size), dtype='f')

calculator = DistanceCalculator('identity')
aligner = Align.PairwiseAligner()
aligner.mode = 'global'

# @profile
def calc_distance(seq_A, seq_B):
    alignment = format(next(aligner.align(seq_A, seq_B))).split('\n')
    aligned_A = SeqRecord(Seq(alignment[0]), id='A')
    aligned_B = SeqRecord(Seq(alignment[-2]), id='B')
    # alignment = pairwise2.align.globalxx(seq_A, seq_B, one_alignment_only=True)[0]
    # aligned_A = SeqRecord(Seq(alignment[0]), id='A')
    # aligned_B = SeqRecord(Seq(alignment[1]), id='B')
    aln = MultipleSeqAlignment([aligned_A, aligned_B])
    dm = calculator.get_distance(aln)
    return 1 - dm.matrix[1][0]

vcalc_distance = np.vectorize(calc_distance, otypes=[float])

# rows_per_rank =
# start_idx = (rank - 1) * rows_per_rank
# stop_idx = None if rank == n_processes - 1 else start_idx + rows_per_rank

rank_rows = itertools.islice(range(size), rank, None, n_processes)
rank_rows_len = len(list(itertools.islice(range(size), rank, None, n_processes)))
print("rank {} starts calcusltions".format(rank), flush=True)
for i, p in enumerate(rank_rows):
    # row_sequence = np.array(sequences[indexes[p]], dtype=np.string_)
    row_sequence = sequences[indexes[p]]
    # column_sequences = list([np.array(sequences[indexes[i]], dtype=np.unicode_) for i in range(p + 1, size)])
    column_sequences = list([sequences[indexes[i]] for i in range(p + 1, size)])
    if len(column_sequences) != 0:
        dset[p, p + 1:] = vcalc_distance(row_sequence, column_sequences)
    if i%10 == 0:
        print("rank {}: completed row {}. {}/{} in {} seconds".format(rank, p, i+1, rank_rows_len, time.time() - start_time), flush=True)


# wait in process rank 0 until process 1 has written to the array
comm.Barrier()


def save_calculated_distance_matrix(data, out_dir, indexes):
    data = data + data.T
    matrix = pd.DataFrame(data, index=indexes, columns=None)
    matrix.columns = matrix.index.values
    matrix.to_csv(out_dir, header=False)
    return matrix

# check that the array is actually shared and process 0 can see
# the changes made in the array by process 1
if comm.rank == 0:
    save_calculated_distance_matrix(dset[:], "/home/mateusz/pca/test/nowy_2/dist2.mat", indexes)
    print("Done in {}".format(time.time() - start_time))

del dset
f.close()