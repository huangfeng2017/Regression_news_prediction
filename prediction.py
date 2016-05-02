__author__ = 'ssarka18'

from sklearn.linear_model import Lasso
from sklearn import cross_validation, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_predict
import os
import time
import seaborn
import matplotlib.mlab as mlab
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn import linear_model

def APE(y_hat, y):
    return ((y_hat - y)/y)*100

def rmse(y_hat, y):
    return ((y_hat**2 - y**2))


if __name__ == "__main__":
    news_df = pd.read_csv("data.csv")

    # Selecting the features manually with a gut guess !!!
    # feature_df = news_df[[' n_tokens_title', ' n_tokens_content', ' global_rate_positive_words', ' is_weekend',
    #                       ' num_hrefs', ' LDA_01', ' LDA_02', ' global_rate_negative_words']]

    #feature_df = news_df[[' n_tokens_title', ' is_weekend', ' num_hrefs', ' LDA_01', ' LDA_02']]
    feature_df = news_df[[' n_tokens_content', ' global_rate_negative_words',
                          ' LDA_01']]


    feature_df_centered = (feature_df - feature_df.mean())
    feature_df_centered[' n_tokens_squared'] = feature_df_centered[' n_tokens_content']**2
    feature_df_centered[' g_neg_squared'] = feature_df_centered[' global_rate_negative_words']**2
    feature_df_centered[' LDA_squared'] = feature_df_centered[' LDA_01']**2

    feature_df_centered[' n_tokens_g_neg'] = feature_df_centered[' n_tokens_content']*feature_df_centered[' global_rate_negative_words']
    feature_df_centered[' g_neg_LDA'] = feature_df_centered[' global_rate_negative_words']*feature_df_centered[' LDA_01']
    feature_df_centered[' LDA_squared_tokens'] = feature_df_centered[' LDA_01']*feature_df_centered[' n_tokens_content']
    feature_df_centered[' all'] = feature_df_centered[' n_tokens_content']\
                                  *feature_df_centered[' global_rate_negative_words']*feature_df_centered[' LDA_01']


    # Create the regression model - second type
    Y = news_df[' shares']/news_df[' timedelta']
    Y = np.asarray(Y)
    X = feature_df_centered
    X = np.asarray(X)

    kf = KFold(len(X), n_folds=10)
    linear_err = 0
    lasso_err = 0
    ridge_err = 0
    r_2 = 0
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        try:
           X_train, X_test = X[train_index], X[test_index]
           Y_train, Y_test = Y[train_index], Y[test_index]
        except:
            print(train_index, test_index)
        # # Train the model using the training sets
        # data_train = pd.DataFrame(np.column_stack([X_train, Y_train]),columns=['x_1','x_2',
        #                                                                  'x_3', 'x_4', 'x_5', 'x_6',
        #                                                                  'x_7', 'x_8', 'y'])
        # data_test = pd.DataFrame(np.column_stack([X_test, Y_test]),columns=['x_1','x_2',
        #                                                                  'x_3', 'x_4', 'x_5', 'x_6',
        #                                                                  'x_7', 'x_8', 'y'])
        #
        # #Initialize predictors to all 15 powers of x
        # predictors=['x_1','x_2','x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8']
        # #predictors.extend(['x_%d'%i for i in range(2,3)])
        #
        # #Define the alpha values to test
        # alpha_lasso = [.01]
        #
        # #Initialize the dataframe to store coefficients
        # col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,9)]
        # ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,1)]
        # coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
        #
        # #Iterate over the 10 alpha values:
        # #for i in range(1):
        # #coef_matrix_lasso.iloc[0,] = lasso_regression(data_train, data_test, predictors, alpha_lasso[0])
        # #coef_matrix_lasso.iloc[0,] = linear_regression(data, 1, models_to_plot)
        # #plt.show()
        # #err += coef_matrix_lasso.iloc[0, 0]


        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        #print(regr.coef_)

        # The mean square error
        linear_err += math.sqrt(np.mean((regr.predict(X_test)-Y_test)**2))
        #err += np.mean(abs(regr.predict(X_test) - Y_test)/Y_test)
        #print(np.mean((regr.predict(X_test)-Y_test)**2))

        alphas = [0.01, 0.02, 0.03]
        regr = linear_model.Lasso()
        scores = [regr.set_params(alpha=alpha).fit(X_train, Y_train).score(X_test, Y_test) for alpha in alphas]
        best_alpha = alphas[scores.index(max(scores))]
        regr.alpha = best_alpha
        regr.fit(X_train, Y_train)
        #print(regr.coef_)

        # The mean square error
        lasso_err += math.sqrt(np.mean((regr.predict(X_test)-Y_test)**2))

        regr = linear_model.Ridge()
        scores = [regr.set_params(alpha=alpha).fit(X_train, Y_train).score(X_test, Y_test) for alpha in alphas]
        best_alpha = alphas[scores.index(max(scores))]
        regr.alpha = best_alpha
        regr.fit(X_train, Y_train)
        #print(regr.coef_)

        # The mean square error
        ridge_err += math.sqrt(np.mean((regr.predict(X_test)-Y_test)**2))

    print(ridge_err/10, lasso_err/10, linear_err/10, '\n')
