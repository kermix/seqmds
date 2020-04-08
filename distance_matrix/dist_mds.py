import nssPCA.data as data
import seqdata.data as sdata
import seqdata.plot as plot
from tools.clustering import kmeans, dbscan, gaussianmixture

import pandas as pd
from sklearn.manifold import MDS


N_COMPONENTS = 2
N_CLUSTERS = 6

matrix = sdata.read_distance_matrix('/home/mateusz/pca/test/nowy_2/dist.mat')
#matrix =  pd.read_csv('/home/mateusz/pca/test/nowy_2/dist2.mat', sep=",", index_col=0, header=None)
#matrix =  pd.read_csv('/home/mateusz/pca/test/aaa/out.mat', sep=",", index_col=0, header=None)
#matrix =  pd.read_csv('/home/mateusz/pca/test/przyciete_15.01/dist.mat', sep=",", index_col=0, header=None)
#matrix =  pd.read_csv('/home/mateusz/pca/test/full_15.01/dist.mat', sep=",", index_col=0, header=None)
matrix.columns = matrix.index.values
num_matrix = data.generate(matrix).values

emb = MDS(n_components=N_COMPONENTS, dissimilarity='precomputed', random_state=1)
transformed = emb.fit_transform(matrix)

result = pd.DataFrame(transformed,
                      index=matrix.index,
                      columns=["".join(("PC", str(i + 1))) for i in range(1, N_COMPONENTS+1)])

y_pred_kmeans, _ = kmeans(transformed, N_CLUSTERS)
y_pred_gausmix, _ = gaussianmixture(transformed, N_CLUSTERS)
y_pred_dbscan, _ = dbscan(transformed, 0.1, 3)
#y_pred_dbscan, _ = dbscan(transformed, 0.4, 3)
#y_pred_dbscan, _ = dbscan(transformed, 0.25, 2)




if N_COMPONENTS == 3:
    plot.plot3d(transformed, y_pred_kmeans, matrix.index, "Kmeans")
    plot.plot3d(transformed, y_pred_dbscan, matrix.index, "DBscan")
    plot.plot3d(transformed, y_pred_gausmix, matrix.index, "Gaussian Mixture")
else:
    plot.plot2d_subplots(transformed, y_pred_kmeans, matrix.index, "Kmeans")
    plot.plot2d_subplots(transformed, y_pred_dbscan, matrix.index, "DBscan")
    plot.plot2d_subplots(transformed, y_pred_gausmix, matrix.index, "Gaussian Mixture")
