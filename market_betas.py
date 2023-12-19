# Import necessary libraries
import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta

class MarketBetas:

    def __init__(self):
        '''
        Import all necessary data and do initial transformations.
        '''

        # Import dfs
        self.index_components = pd.read_csv("index_components.csv")
        self.daily_data = pd.read_csv("gvkey_daily.csv")

        # Calculate Adjusted Closing Price
        self.daily_data['mkt_cap'] = self.daily_data.prccd * self.daily_data.cshoc
        self.daily_data['adj_price'] = self.daily_data.prccd / self.daily_data.ajexdi

        # Convert the date column to datetime format
        self.daily_data['datadate'] = pd.to_datetime(self.daily_data['datadate'])
        self.index_components['datadate'] = pd.to_datetime(self.index_components['datadate'])

        # Create yearmonth columns to ease the merging of both dfs
        self.index_components['yearmonth'] = self.index_components['datadate'].dt.to_period('M')
        self.daily_data['yearmonth'] = self.daily_data['datadate'].dt.to_period('M')

        # Group the data by the 'value' (stock) column
        grouped = self.daily_data.groupby('tic')

        # Define a function to calculate returns for each group
        def calculate_returns(group):
            returns = {'ret': group['adj_price'].pct_change()}
            return pd.DataFrame(returns)

        # Apply the function to each group
        rets = calculate_returns(grouped)

        # Combine the original DataFrame 'comp_info' with the calculated returns
        self.daily_data = pd.concat([self.daily_data, rets], axis=1)


    def index_daily_returns(self):
        '''
        Calculate index daily returns.
        '''

        # Merge data necessary to get index daily returns
        self.index_daily = pd.merge(self.index_components, self.daily_data, on=['yearmonth', 'gvkey'], how='left')

        # Remove from the duplicates the ones that don't have matching tickers
        self.index_daily = self.index_daily[~((self.index_daily.duplicated(subset=['gvkey', 'datadate_x'], keep=False)) & (self.index_daily.tic_x != self.index_daily.tic_y))]

        # Calculate the total market cap for each day
        self.index_daily['total_market_cap'] = self.index_daily.groupby('datadate_y')['mkt_cap'].transform('sum')

        # Calculate the weight of each company's market cap
        self.index_daily['weight'] = self.index_daily['mkt_cap'] / self.index_daily['total_market_cap']

        # Calculate the weighted return for each company
        self.index_daily['weighted_return'] = self.index_daily['ret'] * self.index_daily['weight']

        # Get daily index returns
        self.index_returns = self.index_daily.groupby('datadate_y')['weighted_return'].sum().reset_index()

        # Rename columns
        self.index_returns.rename({'datadate_y':'datadate', 'weighted_return':'index_return'}, axis=1, inplace=True)

    
    def calculate_beta(self, group):
        '''
        Function to calculate correlation between individual stock and index.
        ---
        group: data referring to only one stock
        '''

        group['market_beta'] = group['ret'].rolling(window=len(group), min_periods=1).corr(group['index_return'])
        return group


    def apply_calculate_betas(self):
        '''
        Apply function calculate_beta to the df.
        '''

        # Keep only components of the index
        self.daily_data = self.daily_data.loc[self.daily_data.gvkey.isin(list(self.index_components.gvkey.unique())), ['datadate', 'yearmonth', 'gvkey', 'ret']]
        self.daily_data.dropna(subset=['ret'], inplace=True)

        merged = pd.merge(self.daily_data, self.index_returns, on='datadate', how='left').dropna()

        # Apply the function to each group of 'gvkey'
        self.beta_df = merged.groupby('gvkey').apply(self.calculate_beta)

        # Extract the desired columns
        self.beta_df = self.beta_df[['gvkey', 'yearmonth', 'market_beta']]

        # Reset index
        self.beta_df = self.beta_df.reset_index(drop=True)

        # Get last observation for each month
        self.beta_df = self.beta_df.groupby(['gvkey', 'yearmonth']).last().reset_index()

        # Add one month to each observation, correlation @ 01-2000 does not include current month (hasn't happened yet)
        self.beta_df['yearmonth'] = self.beta_df['yearmonth'] + 1

        return self.beta_df
    

    def get_data(self):
        '''
        Apply all functions to get the desired df.
        '''

        self.index_daily_returns()
        self.apply_calculate_betas()

        return self.beta_df
    

    def write_file(self):
        '''
        Extract dataframes to csv format.
        '''
        
        # Extract dataframe to csv format
        self.beta_df.to_csv('market_betas.csv', index=False)


