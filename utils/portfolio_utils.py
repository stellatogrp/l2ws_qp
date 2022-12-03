from fileinput import filename
import numpy as np
import pandas as pd
import csv
import quandl
import pdb
import datetime as dt
import os
import time
# quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']


def main():
    new()

def new():
    stacked_filename = 'data/portfolio_data/WIKI_prices_all.csv'
    stacked_df = pd.read_csv(stacked_filename)

    # create a new dataframe with 'date' column as index
    new = stacked_df.set_index('date')

    # use pandas pivot function to sort adj_close by tickers
    clean_data = new.pivot(columns='ticker')

    # check the head of the output
    clean_data.head()

    clean_data_shorter = clean_data.iloc[2300:, :]

    # fill in missing entries
    ret = clean_data_shorter.diff() / clean_data_shorter
    ret = ret.fillna(ret.mean())

    short_ret = ret.iloc[:, :3000]
    short_ret.to_csv('data/portfolio_data/returns.csv')

    covariance = short_ret.cov()
    covariance.to_csv('data/portfolio_data/covariance.csv')

    short_ret_np = short_ret.to_numpy()
    cov_np = covariance.to_numpy()
    filename = 'data/portfolio_data/ret_cov.npz'
    np.savez(filename, ret=short_ret_np, cov=cov_np)
    

def old():
    '''
    read csv file that contains all the headers
    '''
    # symbol_filename = 'data/portfolio_data/WIKI_prices.csv'
    # symbol_df = pd.read_csv(symbol_filename)
    stacked_filename = 'data/portfolio_data/WIKI_prices_all.csv'
    stacked_df = pd.read_csv(stacked_filename)

    dates = stacked_df['date'].unique()
    dates = np.insert(dates, 0, 'ticker')
    out_df = pd.DataFrame(columns=dates)
    print('num_tickers', len(symbol_df.ticker))
    for ticker in symbol_df.ticker:
        print('ticker', ticker)
        # filtered = stacked_df.filter(like=ticker, axis=0)
        curr = stacked_df.ticker == ticker
        vals = stacked_df[curr]
        new_row = dict(zip(vals['date'], vals['adj_close']))
        new_row['ticker'] = ticker
        out_df = out_df.append(new_row, ignore_index=True)

        # pdb.set_trace()
    out_df.to_csv('data/portfolio_data/all_stocks.csv')
    # pdb.set_trace()
    # return
    '''
    # make quandl API call
    # '''
    # start_date = dt.datetime(2016, 1, 1)
    # end_date = dt.datetime(2016, 2, 2)
    # raw_tickers = symbol_df.ticker[:1000]
    # tickers = "WIKI/" + raw_tickers.str.replace(".", "_")
    # partial_request = list(tickers + ".11") #+ list(tickers + ".12")
    # request_field = list(sorted(partial_request))

    # print("requesting {} tickers".format(len(request_field)))
    # t0 = time.time()
    # raw_s_data = quandl.get(request_field,
    #                         start_date=start_date, end_date=end_date)

    # print("processing data...")
    # print(f"request took {time.time() - t0} seconds")

    # '''
    # write to a new csv file
    # each ticker will have 13 years of stock data
    # (3000) * (13*365)
    # '''
    # # pdb.set_trace()
    # # df_results = pd.concat(raw_s_data, ignore_index=True).sort_values(
    # #     by=['col_1', 'col6']).reset_index(drop=True)

    # # # Store dataframe at each iteration (in case something breaks)
    # # df_results.to_csv('myfile.csv')
    # raw_s_data.to_csv('data/portfolio_data/stock_prices.csv')


if __name__ == '__main__':
    main()
