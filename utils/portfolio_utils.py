from fileinput import filename
import numpy as np
import pandas as pd
import csv
import quandl
import pdb
import datetime as dt
import os
import time
from get_all_tickers import get_tickers as gt
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
# quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']


def main():
    nasdaq()
    # yahoo()

def nasdaq():
    # stacked_filename = 'data/portfolio_data/WIKI_prices_all.csv'
    stacked_filename = 'data/portfolio_data/EOD.csv'
    stacked_df = pd.read_csv(stacked_filename)

    # create a new dataframe with 'date' column as index
    new = stacked_df.set_index('date')

    # use pandas pivot function to sort adj_close by tickers
    clean_data = new.pivot(columns='ticker')

    # check the head of the output
    clean_data.head()

    close_data = clean_data.dropna(axis=1, how='any')
    close_data_shorter = close_data.iloc[:3000, :]
    

    # clean_data_shorter = clean_data.iloc[2300:, :]

    # fill in missing entries
    ret = close_data_shorter.diff() / close_data_shorter
    ret = ret.fillna(ret.mean())

    short_ret = ret.iloc[:, :3000]
    short_ret = short_ret.clip(lower=-.5, upper=.5)
    pdb.set_trace()
    short_ret.to_csv('data/portfolio_data/returns.csv')

    covariance = short_ret.cov()
    covariance.to_csv('data/portfolio_data/covariance.csv')

    short_ret_np = short_ret.to_numpy()
    cov_np = covariance.to_numpy()
    # cov_np[cov_np > .001] = .001
    # cov_np[cov_np < -.001] = -.001
    filename = 'data/portfolio_data/eod_ret_cov.npz'
    pdb.set_trace()
    np.savez(filename, ret=short_ret_np, cov=cov_np)
    
def yahoo():
    '''
    get all tickers
    '''
    tickers_df = pd.read_csv('data/portfolio_data/yahoo_tickers.csv')
    tickers_list_ = tickers_df.values.tolist()
    tickers_list = [tt[0] for tt in tickers_list_]
    data1 = yf.download(tickers_list[:2000], start="2020-01-01", end="2021-01-01")
    data2 = yf.download(tickers_list[2000:], start="2020-01-01", end="2021-01-01")
    data = data1.append(data2, ignore_index=True)
    close_data_all = data['Adj Close']
    close_data = close_data_all.dropna(axis=1, how='all')
    close_data = close_data.fillna(close_data.mean())
    
    ret = close_data.diff() / close_data
    covariance = ret.cov()

    ret_np = ret.to_numpy()
    cov_np = covariance.to_numpy()
    filename = 'data/portfolio_data/yahoo_ret_cov.npz'

    np.savez(filename, ret=ret_np, cov=cov_np)


if __name__ == '__main__':
    main()
