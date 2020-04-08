import nssPCA.preprocessing as preprocessing
import nssPCA.decomposition as decomposition
import nssPCA.data as data
import seqdata.data as sdata
import seqdata.plot as plot
from tools.clustering import kmeans, dbscan, gaussianmixture

import pandas as pd
import numpy as np

N_CLUSTERS = 6

matrix = sdata.read_distance_matrix('/home/mateusz/pca/test/nowy/dist.mat')
num_matrix = data.generate(matrix).values

scaler = preprocessing.Scaler(calc_mean=True, calc_std=False)

std_matrix = scaler.transform(num_matrix)
std_matrix = pd.DataFrame(std_matrix, index=matrix.index, columns=matrix.columns)

pca = decomposition.SVDecomposition(axis=1)
pca.fit(std_matrix)

with np.printoptions(precision=5, suppress=True):
    print("Ilość wyjaśnianej wariancji przez kolejne zmienne {}".format(pca.explained_ratio))

n_components = input("Ilu głównych składowych użyć do rzutowania? [{}]".format(len(pca.components))) \
               or str(len(pca.components))

pca.number_of_components = int(n_components)

transformed = pca.transform(std_matrix)

result = pd.DataFrame(transformed,
                      index=std_matrix.index,
                      columns=["".join(("PC", str(i + 1))) for i in range(int(n_components))])

y_pred_kmeans, _ = kmeans(transformed, N_CLUSTERS)
y_pred_gausmix, _ = gaussianmixture(transformed, N_CLUSTERS)
y_pred_dbscan, _ = dbscan(transformed, 0.1, 3)

for y_pred in (y_pred_kmeans, y_pred_dbscan, y_pred_gausmix):
    if pca.number_of_components == 3:
        plot.plot3d(transformed, y_pred, matrix.index)
    else:
        plot.plot2d_subplots(transformed, y_pred, matrix.index)


# from sklearn.decomposition import TruncatedSVD
#
# columns = ["".join(("PC", str(i))) for i in range(1, int(len(pca.explained_ratio)) + 1)]
# pd.DataFrame(pca.eigen_values, index=columns).to_csv('/home/mateusz/pca/output/eigenvalues.csv')
#
# columns = std_matrix.columns
# index = ["".join(("PC", str(i))) for i in range(1, int(len(pca.components)) + 1)]
# pd.DataFrame(pca.components, index=index, columns=columns).to_csv('/home/mateusz/pca/output/eigenvectors.csv')
#
# save_important_seqs('/home/mateusz/pca/output/eigenvectors.csv', '/home/mateusz/pca/output/important_seqs.csv', n_components, .15)
#
# sequences = read_seq_data('/home/mateusz/pca/aligned.fa')
# important_sequences = np.genfromtxt('/home/mateusz/pca/output/important_seqs.csv', delimiter=',', dtype=str)
# sequences = sequences.loc[important_sequences, :]
#
# sequences_as_bool_vec = data_to_bool_vec(sequences)
#
# centered = center_bool_vec(sequences_as_bool_vec)
#
# svd = TruncatedSVD(min(centered.shape)-1, algorithm='arpack')
#
# svd.fit(centered)
#
# np.savetxt('/home/mateusz/pca/output/2_explained_var_ratio.csv', svd.explained_variance_ratio_)
#
# eigenvectors = pd.DataFrame(svd.components_, index=["".join(['PC', str(i)]) for i in range(min(centered.shape)-1)], columns=sequences_as_bool_vec.columns)
#
# (eigenvectors.loc[:, (eigenvectors != 0).any(axis=0)]).to_csv('/home/mateusz/pca/output/results.csv')
