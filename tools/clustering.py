from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def kmeans(data, n_clusters):
    engine = KMeans(n_clusters=n_clusters)
    y_pred = engine.fit_predict(data)
    return y_pred, engine


def dbscan(data, eps, min_samples):
    engine = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = engine.fit_predict(data)
    return y_pred, engine


def gaussianmixture(data, n_clusters):
    engine = GaussianMixture(n_components=n_clusters)
    y_pred = engine.fit_predict(data)
    return y_pred, engine
