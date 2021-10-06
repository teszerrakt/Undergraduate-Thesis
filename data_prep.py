import pandas as pd
import numpy as np


def data_split(fold=10, input="ratings.csv", output="./"):
    df = pd.read_csv(input)
    amount = int(len(df)/fold)

    i = 1

    while i <= fold:

        df = df.reindex(np.random.permutation(df.index))
        train = df[amount:]
        test = df[:amount]

        train.to_csv(output + "/train" + "/train" +
                     str(i) + ".csv", index=False)
        test.to_csv(output + "/test" + "/test" + str(i) + ".csv", index=False)

        i += 1


def scale_rating(rating, min_rating=1, max_rating=5):
    # Scale the rating to make the value range from 0-1
    scaled_rating = rating.copy()

    for i in range(min_rating, max_rating+1):
        scaled_rating['rating'] = scaled_rating['rating'].replace(
            [i], i/max_rating)

    return scaled_rating


def load_rating_data(data_type="train", number=1, rating_path="./dataset/", item_path="comic_genre.csv"):
    # load split dataset
    df = pd.read_csv(rating_path + data_type + "/" +
                     data_type + str(number) + ".csv")

    if data_type == "train":
        user_item_matrix = df.pivot_table(
            index='comicID', columns='username', values='rating', fill_value=0, aggfunc='max')
        # scaled_rating = scale_rating(df)
        scaled_rating = scale_rating(df, 1, 5)
        item = pd.read_csv(item_path)

        cluster_dataset = scaled_rating.pivot_table(
            index='comicID', columns='username', values='rating', fill_value=0, aggfunc='max')
        cluster_dataset = pd.merge(cluster_dataset, item, on='comicID')

        return user_item_matrix, cluster_dataset.iloc[:, 1:]

    elif data_type == "test":
        user_item_matrix = df.pivot_table(
            index='comicID', columns='username', values='rating', aggfunc='max')
        user_item_matrix = {int(k1): {k: float(v) for k, v in v1.items() if not np.isnan(v)}
                            for k1, v1 in user_item_matrix.to_dict(orient="index").items()}

        return user_item_matrix
