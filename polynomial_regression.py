__author__ = 'ssarka18'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
import numpy as np
import statsmodels.api as sm
import statsmodels
from pylab import *
import seaborn as sns

p_value_criterion = 0.05

if __name__ == "__main__":
    news_df = pd.read_csv("data.csv")

    #Some scatter plot fo visualizing the data
    # plt.scatter(news_df[' timedelta'], news_df[' shares'])
    # plt.ylim([0, 200000])
    # plt.title('Number of shares vs Number of days distribution', fontsize=20)
    # plt.xlabel('Number of days since post', fontsize=20)
    # plt.ylabel('Number of shares', fontsize=20)
    # plt.show()

    # Regression code starts from here

    # STEP 1: We implement the backward stepwise regression procedure to select features.

    # Selecting the features manually with a gut guess !!!
    feature_df = news_df[[' n_tokens_title', ' num_imgs',
                          ' LDA_01']]


    feature_df_centered = (feature_df - feature_df.mean())
    feature_df_centered[' n_tokens_squared'] = feature_df_centered[' n_tokens_title']**2
    feature_df_centered[' num_imgs_squared'] = feature_df_centered[' num_imgs']**2
    feature_df_centered[' LDA_squared'] = feature_df_centered[' LDA_01']**2

    feature_df_centered[' n_tokens_imgs'] = feature_df_centered[' n_tokens_title']*feature_df_centered[' num_imgs']
    feature_df_centered[' num_imgs_LDA'] = feature_df_centered[' num_imgs']*feature_df_centered[' LDA_01']
    feature_df_centered[' LDA_squared_tokens'] = feature_df_centered[' LDA_01']*feature_df_centered[' n_tokens_title']
    feature_df_centered[' all'] = feature_df_centered[' n_tokens_title']\
                                  *feature_df_centered[' num_imgs']*feature_df_centered[' LDA_01']

    # Create the regression model - second type
    Y = news_df[' shares']/news_df[' timedelta']
    X = feature_df_centered
    p_values = np.ones(15)
    iter_num = 0
    X = sm.add_constant((X))

    if True:
        #print("Iteration: ", iter_num+1)
        model = sm.OLS(Y,X)
        results = sm.OLS(Y, X).fit()
        normalized_cov = results.normalized_cov_params
        params = results.params
        sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
        print(results.summary())
        model_sse = sum.ess
        print('Statistics of errors: ')
        print('centered sum of errors: ', sum.ess)
        print('sum of squared residuals: ', sum.ssr)

    # predicted_values = results.predict()
    # plt.figure()
    # plt.plot(X, Y, 'o')
    # plt.plot(X, predicted_values, 'r--.')
    # # plt.plot(x, iv_u, 'r--')
    # # plt.plot(x, iv_l, 'r--')
    # plt.title('blue: true,   red: OLS')
    # plt.show()

    # features_name = []
    # sse_list = []
    # num_feat_list = range(10)
    # for feat in feature_df:
    #     print('Model without feature: ', feat)
    #     X_feat = X[feat]
    #     X = X.drop(feat, 1)
    #     model = sm.OLS(Y,X)
    #     results = sm.OLS(Y, X).fit()
    #     normalized_cov = results.normalized_cov_params
    #     params = results.params
    #     sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
    #     #print(results.summary())
    #     print('Statistics of errors: ')
    #     print('centered sum of errors: ', sum.ess)
    #     print('sum of squared residuals: ', sum.ssr)
    #     feat_without_sse = sum.ess
    #     ssr_feat_given_model = feat_without_sse - model_sse
    #     print('SSr(',feat,'|x1,...,xn): ', ssr_feat_given_model)
    #     r_1_23 = ssr_feat_given_model/model_sse
    #     print("Coeff. of partial determination(X1|X2,X3): ", r_1_23)
    #
    #     model = sm.OLS(Y,X_feat)
    #     results = sm.OLS(Y, X_feat).fit()
    #     normalized_cov = results.normalized_cov_params
    #     params = results.params
    #     sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
    #     r_1 = sum.ssr / (sum.ssr + sum.ess)
    #     print("Coeff. of partial determination(X1): ", r_1)
    #     print("\n")
    #     X = feature_df
    #
    #
    #
    #

