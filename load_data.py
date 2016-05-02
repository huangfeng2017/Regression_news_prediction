__author__ = 'ssarka18'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
import numpy as np
import statsmodels.api as sm
import statsmodels

p_value_criterion = 0.05


def check_pvalues(values_list):
    for i in values_list:
        if i >= p_value_criterion:
            return True
    return False

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
    feature_df = news_df[[' n_tokens_title', ' num_imgs', ' n_unique_tokens', ' self_reference_avg_sharess', ' LDA_04',
                          ' title_sentiment_polarity', ' global_subjectivity', ' global_rate_positive_words', ' is_weekend',
                          ' num_hrefs', ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' global_rate_negative_words']]

    # Create the regression model - first type
    # Add  column of 1's array to the dataframe
    # values = np.ones(len(news_df, ))
    # ones_col = pd.DataFrame({'ones': values})
    # feature_df = feature_df.join(ones_col)
    # Y = news_df[' shares']
    # X = feature_df
    # result = smf.OLS(Y, X ).fit()
    #print(result.summary())

    # Create the regression model - second type
    Y = news_df[' shares']
    X = feature_df
    p_values = np.ones(15)
    iter_num = 0
    X = sm.add_constant((X))

    while (len(p_values) > 1) and check_pvalues(p_values):
        print("Iteration: ", iter_num+1)
        model = sm.OLS(Y,X)
        results = sm.OLS(Y, X).fit()
        normalized_cov = results.normalized_cov_params
        params = results.params
        sum = statsmodels.regression.linear_model.RegressionResults(model, params, normalized_cov)
        p_values = sum.pvalues
        p_values = p_values[1:]
        print("P values: ", p_values)
        max_index = np.argmax(p_values)
        column_headers = list(X.columns.values)
        feat_drop = column_headers[max_index+1]
        print("Deleting feature ", feat_drop, " from the list...\n")
        X = X.drop(feat_drop, 1)
        p_values = np.delete(p_values, max_index)
        iter_num += 1

    print(list(X.columns.values))






