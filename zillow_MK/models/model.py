import os.path
import time
import numpy as np

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA


import zillow_MK.config as config
from zillow_MK.config import MODELS_SUBDIR
from zillow_MK.utils import utils
from zillow_MK.utils.utils import save_binary, load_binary, prepare_data, evaluate_prediction,\
    report


class Model:
    def __init__(self, RF: bool = True):
        self.vectorizer = DictVectorizer()
        self.imputer = Imputer()
        self.scaler = MaxAbsScaler()
        self.clf = RandomForestRegressor(random_state=11) if RF else XGBRegressor()
        self.pca = PCA(n_components=20)
        self.RF = RF

        model_version = 'RF' if RF else 'XGB'
        self.model_path = os.path.join(MODELS_SUBDIR, f'{model_version}.clf')
        self.vectorizer_path = os.path.join(MODELS_SUBDIR, f'vectorizer_{model_version}.clf')
        self.imputer_path = os.path.join(MODELS_SUBDIR, f'imputer_{model_version}.clf')
        self.scaler_path = os.path.join(MODELS_SUBDIR, f'scaler_{model_version}.clf')

    def process_train_data(self, data_train):
        X_train = self.vectorizer.fit_transform(data_train)
        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)

        print(X_train.shape)

        return X_train.toarray()

    def process_test_data(self, data_test):
        X_test = self.vectorizer.transform(data_test)
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)

        return X_test.toarray()

    def param_tuning(self, data_train, y_train, perform_pca: bool = False):
        X_train = self.process_train_data(data_train)

        if perform_pca:
            X_train = self.pca.fit_transform(X_train)

        search_params = {}

        if self.RF:

            search_params = {
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]
            }

        else:
            search_params = {
                'learning_rate': [0.02, 0.04],
                'min_child_weight': [30, 60],
                'max_depth': [3, 5, 7],
                'subsample': [0.4, 0.2],
                'n_estimators': [100, 200, 300],
                'colsample_bytree': [0.6, 0.4],
                'reg_lambda': [1, 2],
                'reg_alpha': [1, 2]
            }

        n_iter_search = 20
        cv = RandomizedSearchCV(self.clf, param_distributions=search_params, n_iter=n_iter_search,
                                scoring={'score': 'neg_mean_absolute_error'}, n_jobs=-1, cv=3,
                                refit='score')

        start = time.time()
        cv.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time.time() - start), n_iter_search))
        print("-----------------------")
        print("BEST PARAMETERS ARE:")
        print(cv.best_params_)
        # report(cv.cv_results_)

        self.clf = cv.best_estimator_

        # by the end of randomized search, the model will be refitted on the entire data set
        print('saving model to file...')
        save_binary(self.clf, self.model_path)

        print('saving vectorizer to file...')
        save_binary(self.vectorizer, self.vectorizer_path)

        print('saving imputer to file...')
        save_binary(self.imputer, self.imputer_path)

        print('saving scaler to file...')
        save_binary(self.scaler, self.scaler_path)

        X_test = self.process_test_data(data_test)

        if perform_pca:
            X_test = self.pca.transform(X_test)

        preds = self.clf.predict(X_test)
        evaluate_prediction(preds, y_test)


if __name__ == '__main__':
    try:
        data_train, data_test, y_train, y_test = load_binary(os.path.join(MODELS_SUBDIR, 'data.dat'))
    except:
        data_train, data_test, y_train, y_test = prepare_data()
        save_binary((data_train, data_test, y_train, y_test), os.path.join(MODELS_SUBDIR, 'data.dat'))

    combinations = [(rf, pca) for rf in (True, False) for pca in (True, False)]
    for rf, pca in combinations:
        str_method = "Random Forest" if rf else "XGB"
        str_pca = "with PCA" if pca else "without PCA"
        print("----------------------------------------------------------")
        print(f'Training %s Regressor %s' % (str_method, str_pca))
        model = Model(RF=rf)
        model.param_tuning(data_train, y_train, perform_pca=pca)
