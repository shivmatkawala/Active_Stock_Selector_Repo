import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from pytz import timezone

"""
symbol = 'RELIANCE.NS'
start_date = '2020-01-01'
end_date = '2022-01-01'

stock_data = yf.download(symbol, start=start_date, end=end_date).tz_localize('UTC').tz_convert('Asia/Kolkata')
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)

print(stock_data.head())
"""


class Stock:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.historical_data = self.download_historical_data()

    def download_historical_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data

    def cur_price(self, cur_date):
        if cur_date in self.historical_data.index:
            return self.historical_data.loc[cur_date, 'Close']
        else:
            # Handle the case where the date is not present in the dataset
            return None

    def n_day_ret(self, N, cur_date):
        cur_price = self.cur_price(cur_date)
        prev_price = self.cur_price(cur_date - pd.DateOffset(days=N))

        if cur_price is not None and prev_price is not None:
            return ((cur_price / prev_price) - 1) * 100
        else:
            # Handle the case where either cur_price or prev_price is None
            return None

    def daily_ret(self, cur_date):
        return self.historical_data.loc[cur_date, 'Close'] / self.historical_data.loc[
            cur_date - pd.DateOffset(days=1), 'Close'] - 1

    def last_30_days_price(self, cur_date):
        return self.historical_data.loc[cur_date - pd.DateOffset(days=30):cur_date, 'Close'].values


def calculate_metrics(equity_curve):
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (1 / len(equity_curve.index.year.unique())) - 1) * 100
    volatility = np.std(equity_curve.pct_change()) * np.sqrt(252) * 100
    sharpe_ratio = (cagr / volatility) if volatility != 0 else 0
    return cagr, volatility, sharpe_ratio


def main():
    # Sample input
    start_date = '2019-01-01'
    end_date = '2021-01-01'
    initial_equity = 1000000

    # List of NIFTY 50 stocks (you can update this list)
    nifty_50_stocks = ['RELIANCE.NS', 'HCLTECH.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS',
                       'BAJFINANCE.NS', 'APOLLOHOSP.NS',
                       'WIPRO.NS', 'ADANIENT.NS']

    # Data for benchmark (NIFTY 50 index)
    benchmark_data = yf.download('^NSEI', start=start_date, end=end_date)

    # Initialize the Streamlit app
    st.title('Stock Selection Strategy Analysis')

    # Display user inputs
    st.write(f"Simulation Start date: {start_date}")
    st.write(f"Simulation End date: {end_date}")
    st.write(f"Initial Equity: {initial_equity}")

    # Store equity curves for benchmark and strategy
    benchmark_equity_curve = pd.Series(initial_equity, index=benchmark_data.index)
    strategy_equity_curve = pd.Series(initial_equity, index=benchmark_data.index)

    # Iterate through each month and update the strategy
    for month_end in pd.date_range(start=start_date, end=end_date, freq='M'):
        selected_stocks = []

        # Update the strategy based on positive returns
        for stock_ticker in nifty_50_stocks:
            stock = Stock(stock_ticker, month_end - pd.DateOffset(months=1), month_end)
            n_day_return = stock.n_day_ret(30, month_end)

            if n_day_return is not None and n_day_return > 0:
                selected_stocks.append(stock_ticker)

        # Update the equity curve for the strategy
        if selected_stocks:
            strategy_daily_ret = sum(
                [Stock(stock_ticker, date - pd.DateOffset(days=1), date).daily_ret(date) for stock_ticker in
                 selected_stocks]) / len(selected_stocks)
            strategy_equity_curve[date] = strategy_equity_curve.index[previous_date] * (1 + strategy_daily_ret)

    # Calculate metrics
    benchmark_cagr, benchmark_volatility, benchmark_sharpe_ratio = calculate_metrics(benchmark_equity_curve)
    strategy_cagr, strategy_volatility, strategy_sharpe_ratio = calculate_metrics(strategy_equity_curve)

    # Display results
    st.write("## Results")
    st.write(f"Benchmark CAGR: {benchmark_cagr:.2f}%")
    st.write(f"Benchmark Volatility: {benchmark_volatility:.2f}%")
    st.write(f"Benchmark Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")

    st.write(f"Strategy CAGR: {strategy_cagr:.2f}%")
    st.write(f"Strategy Volatility: {strategy_volatility:.2f}%")
    st.write(f"Strategy Sharpe Ratio: {strategy_sharpe_ratio:.2f}")

    # Plot equity curves
    st.write("## Equity Curves")
    st.line_chart(pd.concat([benchmark_equity_curve, strategy_equity_curve], axis=1, keys=['Benchmark', 'Strategy']))

    # Display selected stocks
    st.write("## Selected Stocks for Sample Strategy")
    st.write(selected_stocks)


if __name__ == '__main__':
    main()
