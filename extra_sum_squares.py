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
    feature_df = news_df[[' n_tokens_title', ' n_tokens_content', ' global_rate_positive_words', ' is_weekend',
                          ' num_hrefs', ' LDA_01', ' LDA_02', ' global_rate_negative_words']]


    # Create the regression model - second type
    Y = news_df[' shares']/news_df[' timedelta']
    X = feature_df
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

        SE = [4.086, 0.219, 0.001, 27.674, 1.369, 0.045, 2.175, 1.74, 43.278]
        i = 0

        for i in range(len(results.params)):
            beta = results.params[i]
            se = SE[i]
            t = beta / se
            #print('t =', t)
            N = results.nobs
            P = results.df_model
            dof = N - P - 1
            hp = 1.0 - scipy.stats.t( dof ).cdf( t )
            p = hp * 2.0
            print(t, scipy.stats.norm.ppf(0.975), t < scipy.stats.norm.ppf(0.975))

    X = feature_df
    features_name = []
    sse_list = []
    num_feat_list = range(8)
    for feat in feature_df:
        print('Model without feature: ', feat)
        X = X.drop(feat, 1)
        model = sm.OLS(Y,X)
        results = sm.OLS(Y, X).fit()
        normalized_cov = results.normalized_cov_params
        params = results.params
        sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
        #print(results.summary())
        print('Statistics of errors: ')
        print('centered sum of errors: ', sum.ess)
        print('sum of squared residuals: ', sum.ssr)
        feat_without_sse = sum.ess
        sse_feat_given_model = feat_without_sse - model_sse
        print('SSr(',feat,'|x1,....xn): ', sse_feat_given_model)
        X = feature_df
        features_name.append(feat)
        sse_list.append(sse_feat_given_model)


    X = feature_df

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    for label, x, y in zip(features_name, num_feat_list, sse_list):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    ax.scatter(num_feat_list, sse_list, color="black", s=30, marker="o")
    #plt.xlabel(r'\textbf{steep interval}', fontsize=50)
    #plt.ylabel(r'\textbf{inhibition interval}',fontsize=50)
    plt.xlabel('Features', fontsize=30)
    plt.ylabel('Extra sum of squares', fontsize=30)
    plt.legend(loc='upper right', fontsize=30)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=30)
    #plt.xticks(num_feat_list, features_name, rotation=30)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.show()

    for feat in feature_df:
        print('Testing null hypothesis with feature: ', feat)
        X = X.drop(feat, 1)
        model = sm.OLS(Y, X)
        results = sm.OLS(Y, X).fit()
        normalized_cov = results.normalized_cov_params
        params = results.params
        sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
        #print(results.summary())
        feat_without_sse = sum.ess
        ssr_feat_given_model = feat_without_sse - model_sse
        f_star = ssr_feat_given_model /(model_sse/(len(news_df.index)-9))
        #print("F_star: ", f_star)
        f_value = 5.02
        print(model_sse, feat_without_sse, f_value, f_star, f_value<f_star)
        X = feature_df


    # Correlation matrix

    # plt.figure()
    # sns.pairplot(feature_df)
    # plt.show()







