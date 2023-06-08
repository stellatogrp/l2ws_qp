import pdb

import numpy as np
import pandas as pd
import sys


def main(data):
    nasdaq(data)


def nasdaq(data):
    if data == 'wiki':
        stacked_filename = 'data/portfolio_data/WIKI_prices_all.csv'
    elif data == 'eod':
        stacked_filename = 'data/portfolio_data/EOD.csv'
    stacked_df = pd.read_csv(stacked_filename)

    # create a new dataframe with 'date' column as index
    new = stacked_df.set_index('date')

    # use pandas pivot function to sort adj_close by tickers
    clean_data = new.pivot(columns='ticker')

    # check the head of the output
    clean_data.head()

    num_nan = clean_data.isna().sum(axis=0)
    # close_data = clean_data.dropna(axis=1, how='any')
    # pdb.set_trace()
    small_nan_data = num_nan < 10
    indices = small_nan_data.to_numpy()
    close_data = clean_data.iloc[:, indices]
    close_data = clean_data

    # now get largest 3000 assets
    sums = close_data.min(axis=0).to_numpy()
    sums_argsorted = np.argsort(sums)
    highest_indices = sums_argsorted[-3000:]

    top3000 = close_data.iloc[:, highest_indices]

    # close_data_shorter = close_data.iloc[:3000, :]
    close_data_shorter = top3000

    # clean_data_shorter = clean_data.iloc[2300:, :]

    # fill in missing entries
    close_data_shorter = close_data_shorter.fillna(close_data_shorter.mean())
    diff = -close_data_shorter.diff()
    diff2 = diff.iloc[1:, :]
    ret = diff2 / close_data_shorter.iloc[:-1, :]
    ret = ret.fillna(ret.mean())

    short_ret = ret.iloc[:, :3000]
    # short_ret = short_ret.clip(lower=-1, upper=1)
    # pdb.set_trace()
    short_ret.to_csv('data/portfolio_data/returns.csv')

    covariance = short_ret.cov()
    covariance.to_csv('data/portfolio_data/covariance.csv')

    short_ret_np = short_ret.to_numpy()
    cov_np = covariance.to_numpy()

    # get factor model
    U, S, VT = np.linalg.svd(cov_np)
    # pdb.set_trace()
    factor = 15
    S_factor = np.diag(S[:factor])
    factor_cov = U[:, :factor] @ S_factor @ VT[:factor, :]

    # cov_np[cov_np > .001] = .001
    # cov_np[cov_np < -.001] = -.001
    if data == 'wiki':
        filename = 'data/portfolio_data/wiki_ret_cov.npz'
    elif data == 'eod':
        filename = 'data/portfolio_data/eod_ret_cov.npz'
    pdb.set_trace()
    np.savez(filename, ret=short_ret_np, cov=factor_cov)


if __name__ == '__main__':
    data = sys.argv[1]
    main(data)
