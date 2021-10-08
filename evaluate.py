from predict import predict
import timeit

def time_evaluate(test_data, ratings_cluster, centroids, verbose=False):
    i = 1
    total_error = 0
    total_time = 0

    for item in test_data:

        for user in test_data[item]:
            # Check if the user exist in training data, if exist calculate
            try:
                start = timeit.default_timer()
                prediction = predict(
                    user, item, ratings_cluster, centroids, k=10)
                end = timeit.default_timer()
                time_ = end-start
                error = test_data[item][user] - prediction

                if verbose:
                    print('{}. item {} | user {} -> {} - {} = {}'.format(i,
                                                                         item, user, test_data[item][user], prediction, error))

                total_time += time_
                total_error += error
                i += 1
            except KeyError:
                continue

    mae = total_error/i
    time = total_time/i

    if verbose:
        print('\nTotal MAE:', str(mae))

    return mae, time
