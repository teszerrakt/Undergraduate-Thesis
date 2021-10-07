import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances as dist


def kmeans_clustering(n_clusters, cluster_data, rating_data):
    # Perform clustering using K-Means clustering to get array of clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                    random_state=1337).fit(cluster_data)
    centroids = kmeans.cluster_centers_
    cluster_prediction = kmeans.predict(cluster_data)

    # Concatenate the array of cluster to the rating dataset
    ratings_cluster = pd.concat([rating_data.reset_index(), pd.DataFrame({
                                'cluster': cluster_prediction})], axis=1)

    # Re-setting the comicID as index
    ratings_cluster = ratings_cluster.set_index('comicID')

    # Return the rating dataset with its cluster number and the centroid coordinate of each cluster
    return ratings_cluster, centroids


def dbscan_clustering(epsilon, cluster_data, rating_data, min_pts=11, verbose=False):
    # Perform clustering using DBSCAN clustering to get array of clusters
    cluster_prediction = DBSCAN(
        eps=epsilon, min_samples=min_pts).fit_predict(cluster_data)

    # Concatenate the array of cluster to the rating dataset
    ratings_cluster = pd.concat([rating_data.reset_index(), pd.DataFrame({
                                'cluster': cluster_prediction})], axis=1)

    if verbose:
        cluster_labels = set(cluster_prediction)
        print('Number of cluster :', len(cluster_labels))
        for c in cluster_labels:
            print('Cluster', c, ':', len(
                ratings_cluster.loc[ratings_cluster['cluster'] == c]), 'titles')

    # Re-setting the comicID as index
    ratings_cluster = ratings_cluster.set_index('comicID')

    # Return the rating dataset with its cluster number
    return ratings_cluster


def find_centroid_distance(which_cluster, centroids, verbose=False):
    centroid_distance = {}

    for i in range(len(centroids)):
        if i == which_cluster:
            continue
        else:
            distance = dist(centroids[which_cluster].reshape(
                1, -1), centroids[i].reshape(1, -1))
            centroid_distance[i] = distance[0][0]

            if verbose:
                print('cluster {} to cluster {} distance: {}'.format(
                    which_cluster, i, centroid_distance[i]))

    return centroid_distance


def merge_cluster(clustered_ratings, all_ratings, which_cluster, centroids, k=10):
    # Substracted by one to remove the input item
    cluster_member = len(clustered_ratings) - 1
    cluster_distance = find_centroid_distance(which_cluster, centroids)
    minimum = min(cluster_distance, key=cluster_distance.get)
    # Delete the minimum cluster from the dictionary
    cluster_distance.pop(minimum)

    while cluster_member < k:
        nearest_cluster = all_ratings.loc[all_ratings['cluster'] == minimum]
        clustered_ratings = pd.concat(
            [clustered_ratings, nearest_cluster], axis=0)
        cluster_member = len(clustered_ratings)-1

    return clustered_ratings
