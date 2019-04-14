import pickle  # for saving/loading binary files (serializing/deserializing)
import time
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def prepare_data() -> (List[str], List[str], List[str], List[str]):
    start = time.time()
    print('Reading training data...')
    train_df = pd.read_csv("../../data/train_2016_v2.csv", parse_dates=["transactiondate"])
    prop = pd.read_csv("../../data/properties_2016.csv",
                       dtype={"hashottuborspa": object, "propertycountylandusecode": object,
                              "propertyzoningdesc": object, "fireplaceflag": object,
                              "taxdelinquencyflag": object})
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))

    df_train_raw = train_df.merge(prop, how='left', on='parcelid').drop(['parcelid','transactiondate',
                                                                     'propertyzoningdesc','taxdelinquencyflag',
                                                                     'propertycountylandusecode', 'hashottuborspa',
                                                                         'fireplaceflag'], axis=1)
    df_train_raw.fillna(df_train_raw.median(), inplace=True)
    df_train = df_train_raw.sample(n=500, random_state=12)
    print(df_train.shape)

    target_col = 'logerror'
    y = []  # the labels
    X = []  # the features
    features = list([x for x in df_train.columns if x != target_col])

    for row in tqdm(df_train.to_dict('records')):
        y.append(row[target_col])
        X.append({k: row[k] for k in features})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


def evaluate_prediction(predictions, y_test):
    print("-----------------------")
    print("MODEL SCORE(MEAN ABSOLUTE ERROR): %.4f" % mean_absolute_error(y_test, predictions))
    #accuracy = accuracy_score(y_test, predictions)
    #print(f'accuracy: {round(accuracy, 3)}')


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))


#Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        # print(results)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
