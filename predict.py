from sklearn.neighbors import NearestNeighbors
from cluster import merge_cluster


def find_neighbor(item_id, ratings, metric='correlation', k=10, verbose=False):
    similarities = []
    indices = []
    ratings = ratings.iloc[:, :-1]  # don't use the cluster columns
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(
        ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()

    if verbose:
        print('{} most similar items for item {}:\n'.format(k, item_id))
        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i]+1 == item_id:
                continue
            else:
                print('{}: {}, with similarity of {}'.format(
                    i, ratings.index[indices.flatten()[i]], similarities.flatten()[i]))

    return similarities, indices


def predict(user_id, item_id, ratings_cluster, centroids, metric='correlation', k=10, verbose=False):
    prediction = wtd_sum = 0

    user_loc = ratings_cluster.columns.get_loc(user_id)

    which_cluster = ratings_cluster.loc[ratings_cluster.index ==
                                        item_id, 'cluster'].iloc[0]
    clustered_ratings = ratings_cluster.loc[ratings_cluster['cluster']
                                            == which_cluster]

    # Check wether the cluster has enough member for neighbor finding process, if not then merge the current cluster with the nearest cluster
    # Substracted by one to remove the input item
    if centroids != None:
        cluster_member = len(clustered_ratings) - 1
        if cluster_member < k:
            clustered_ratings = merge_cluster(
                clustered_ratings, ratings_cluster, which_cluster, centroids, k=k)

    item_loc = clustered_ratings.index.get_loc(item_id)

    similarities, indices = find_neighbor(
        item_id, clustered_ratings, metric=metric, k=k)

    sum_wt = 0
    product = 1

    for i in range(0, len(indices.flatten())):
        # if the index == item to be predicted or rating == 0 skip
        if (indices.flatten()[i] == item_loc) or (clustered_ratings.iloc[indices.flatten()[i], user_loc] == 0):
            continue
        else:
            # rating * similarity
            product = clustered_ratings.iloc[indices.flatten()[
                i], user_loc] * (similarities[i])

            # sum of similarity
            sum_wt += abs(similarities[i])

            if verbose:
                print('{}. item_loc: {}, user_loc: {} -> rating: {} * similarity: {} = {}'.format(i, indices.flatten()
                      [i], user_loc, clustered_ratings.iloc[indices.flatten()[i], user_loc], similarities[i], product))

            wtd_sum += product

    if sum_wt == 0:
        prediction = 0  # if the sum of similarity == 0, then the predicted rating is also 0
    else:
        prediction = wtd_sum/sum_wt

    if prediction < 1:
        prediction = 1
    elif prediction > 5:
        prediction = 5

    if verbose:
        print(
            'Predicted rating for user {} -> item {}: {}'.format(user_id, item_id, prediction))

    return prediction
