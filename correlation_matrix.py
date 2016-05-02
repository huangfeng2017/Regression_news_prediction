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
def plot_corr(df, size=40):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tick_params('x', labelsize=10)
    plt.tick_params('y', labelsize=30)
    plt.show()

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
    plot_corr(feature_df)
