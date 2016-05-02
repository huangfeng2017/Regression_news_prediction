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


p_value_criterion = 0.05

if __name__ == "__main__":
    news_df = pd.read_csv("data.csv")

    # Selecting the features manually with a gut guess !!!
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


    plt.hist(sum.resid, bins=40)
    plt.ylabel('Count', fontsize=30)
    plt.xlabel('Residuals', fontsize=30)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=20)
    #plt.xticks(num_feat_list, features_name, rotation=30)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.show()

    cnt = 1
    for feat in X:
        fig = plt.figure(figsize=(10, 8))
        #ax = fig.add_subplot(1,1,cnt) # one row, one column, first plot
        plt.subplot(111)
        plt.scatter(X[feat], sum.resid, color="black", s=30, marker="o")
        #plt.xlabel(r'\textbf{steep interval}', fontsize=50)
        #plt.ylabel(r'\textbf{inhibition interval}',fontsize=50)
        plt.xlabel(feat, fontsize=30)
        plt.ylabel('Residuals', fontsize=30)
        plt.legend(loc='upper right', fontsize=30)
        plt.tick_params('x', labelsize=20)
        plt.tick_params('y', labelsize=20)
        #plt.xticks(num_feat_list, features_name, rotation=30)
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.show()
        cnt += 1
