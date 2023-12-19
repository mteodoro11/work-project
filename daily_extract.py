# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch 
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class DailyData:
    

    def __init__(self):
        '''
        Import all necessary data and do initial transformations.
        '''

        # Get financial ratios
        self.index_components = pd.read_csv("index_components.csv")
        # Transform datadate to datetime
        self.index_components['datadate'] = pd.to_datetime(self.index_components['datadate'])

        # Get daily data (csv from WRDS)
        self.daily_data = pd.read_csv("gvkey_daily.csv")
        # Calculate Market Cap as Price @ Close unadjusted x Shares outstanding
        self.daily_data['mkt_cap'] = self.daily_data.prccd * self.daily_data.cshoc
        # Calculate Adjusted Closing Price
        self.daily_data['adj_price'] = self.daily_data.prccd / self.daily_data.ajexdi
        # Convert the date column to datetime format
        self.daily_data['datadate'] = pd.to_datetime(self.daily_data['datadate'])
        # Calculate daily returns
        self.calculate_daily_returns()
        # Drop rows where values are missing from important columns
        self.daily_data = self.daily_data.dropna(subset=['cshoc'])
        self.daily_data = self.daily_data.dropna(subset=['ret'])
        # Subset data to where prices and mkt caps start existing in the data
        self.daily_data = self.daily_data[self.daily_data.datadate>="1998-04-01"]

        # Extract the year and month and create a new column
        self.index_components['yearmonth'] = self.index_components['datadate'].dt.to_period('M')
        self.daily_data['yearmonth'] = self.daily_data['datadate'].dt.to_period('M')


    def calculate_daily_returns(self):
        '''
        Calculate daily returns for individual stocks.
        '''

        # Group the data by the 'tic' (stock) column
        grouped = self.daily_data.groupby('tic')

        # Define a function to calculate returns for each group
        def calculate_returns(group):
            returns = {'ret': group['adj_price'].pct_change()}
            return pd.DataFrame(returns)

        # Apply the function to each group
        rets = calculate_returns(grouped)

        # Combine the original DataFrame 'comp_info' with the calculated returns
        self.daily_data = pd.concat([self.daily_data, rets], axis=1)

    
    def extract_prev_vol(self):
        '''
        Extract previous month volatility for each index component.
        '''

        # Create a copy to shift to previous month
        self.prev_month_components = self.index_components[["gvkey", "datadate", "tic", "yearmonth"]].copy()
        self.prev_month_components["yearmonth"] = self.prev_month_components["yearmonth"] - 1

        # Merge daily data to keep only index components
        self.prev_month_data = pd.merge(self.prev_month_components, self.daily_data[["gvkey", "datadate", "tic", "mkt_cap", "ret", "yearmonth"]], on=['yearmonth', 'gvkey'], how='inner')
        self.prev_month_data.rename({"datadate_x":"datadate_holding", "datadate_y":"datadate_formation"}, axis=1, inplace=True)

        # Get volatility
        self.prev_month_vol = self.prev_month_data.groupby(['gvkey', 'yearmonth'])['ret'].std().reset_index().rename({"ret":"vol_1m"},axis=1)
        self.prev_month_vol['vol_1m'] = self.prev_month_vol['vol_1m'] * np.sqrt(22)

        # Re-adjust date to holding month
        self.prev_month_vol['yearmonth'] = self.prev_month_vol['yearmonth'] + 1

        # Transform to datadate so it matches everything else
        self.prev_month_vol['yearmonth'] = pd.PeriodIndex(self.prev_month_vol['yearmonth'], freq='M').to_timestamp()
        self.prev_month_vol.rename({"yearmonth":"datadate"}, axis=1, inplace=True)

    
    def get_garch_predictions(self):
        '''
        Apply the GARCH(1,1) model to predict index monthly volatility based on index daily returns
        '''

        # Match daily data to keep only stocks in the index
        self.daily_data = pd.merge(self.index_components[["gvkey", "datadate", "tic", "yearmonth"]], self.daily_data[["gvkey", "datadate", "tic", "mkt_cap", "ret","yearmonth"]], on=['yearmonth', 'gvkey'], how='inner')

        # Remove from the duplicates the ones that don't have matching tickers
        self.daily_data = self.daily_data[~((self.daily_data.duplicated(subset=['gvkey', 'datadate_y'], keep=False)) & (self.daily_data.tic_x != self.daily_data.tic_y))]    
        self.daily_data.rename({'mkt_cap_y': 'mkt_cap', 'ret_y': 'ret'}, axis=1, inplace=True)

        # Calculate the total market cap for each day
        self.daily_data['total_market_cap'] = self.daily_data.groupby('datadate_y')['mkt_cap'].transform('sum')

        # Calculate the weight of each company's market cap
        self.daily_data['weight'] = self.daily_data['mkt_cap'] / self.daily_data['total_market_cap']

        # Calculate the weighted return for each company
        self.daily_data['weighted_return'] = self.daily_data['ret'] * self.daily_data['weight']

        # Group by date and calculate the sum of weighted returns for each day
        grouped_data = self.daily_data.groupby('datadate_y')['weighted_return'].sum().reset_index()

        # Change datadate name
        grouped_data.rename({'datadate_y': 'datadate'}, axis=1, inplace=True)

        # Form training set
        train_data = grouped_data[grouped_data.datadate < "2000-01-01"]
        # Form test set
        test_data = grouped_data[(grouped_data.datadate >= "2000-01-01") & (grouped_data.datadate < "2023-01-01")]

        # Get weighted index returns for studied time period and re-scale them (better for model fit)
        returns = grouped_data[grouped_data.datadate < "2023-01-01"]
        returns = returns['weighted_return']*100

        # Create a list to store predicted volatility
        rolling_predictions = []
        # Set test size
        test_size = len(test_data)

        # Loop to get GARCH volatility predictions for the index on a rolling basis
        for i in range(test_size):
            train = returns[:-(test_size-i)]
            model = arch_model(train, p=1, q=1)
            model_fit = model.fit(disp="off")
            pred = model_fit.forecast(horizon=22, reindex=False)
            rolling_predictions.append(np.average(np.sqrt(pred.variance.values))*np.sqrt(22))

        # Add predictions to test set
        test_data['volatility_predicted'] = [x/100 for x in rolling_predictions]

        # Get index actual volatility for each month
        actual_monthly_volatility = test_data.set_index("datadate").weighted_return.resample('M').std()*np.sqrt(22)

        # Get index previous month volatility
        lagged_monthly_vol = actual_monthly_volatility.shift(1)
        lagged_monthly_vol[0] = (train_data.set_index("datadate").weighted_return.resample('M').std()*np.sqrt(22))[-1]

        # Copy original df
        self.pred_vols = test_data.copy()
        # Set date as the index
        self.pred_vols.set_index('datadate', inplace=True)
        # Drop returns column
        self.pred_vols.drop("weighted_return", axis=1, inplace=True)
        # Keep only the first observation of each month
        self.pred_vols = self.pred_vols.groupby(self.pred_vols.index.to_period("M")).first()
        # Join in the same df the predicted, the actual and the previous month volatility
        self.pred_vols['volatility_actual'] = actual_monthly_volatility.values
        self.pred_vols['volatility_1m'] = lagged_monthly_vol.values
        self.pred_vols.reset_index(inplace=True)


    def extract_monthly_vol(self):
        '''
        Extract monthly volatility for each index component.
        '''

        # Get count of stocks daily observations per month
        observation_counts = self.daily_data.groupby(['gvkey', 'yearmonth']).size().reset_index(name='count')
        # Find stocks where there are less than 5 daily observations per month
        missing = observation_counts[observation_counts['count'] < 5].sort_values(by="yearmonth")[['gvkey', 'yearmonth']]

        # Merge dataframes on 'gvkey' and 'yearmonth'
        merged_df = pd.merge(self.daily_data, missing, on=['gvkey', 'yearmonth'], how='left', indicator=True)

        # Filter out the rows that are common in both dataframes
        self.daily_data = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)

        # Get average daily standard deviation of returns
        self.monthly_vol = self.daily_data.groupby(['gvkey', 'datadate_x'])['ret'].std().reset_index().rename({'ret':'vol'}, axis=1)
        # Calculate monthly volatility
        self.monthly_vol['vol'] = self.monthly_vol['vol'] * np.sqrt(22)


    def get_data(self):
        '''
        Apply all functions to get the desired df.
        '''

        self.extract_prev_vol()
        self.get_garch_predictions()
        self.extract_monthly_vol()
    

    def write_files(self):
        '''
        Extract dataframes to csv format.
        '''

        # Form data dfs
        self.get_data()

        # Write file with previous month volatility
        self.prev_month_vol.to_csv("vol_previous_month.csv", index=False)

        # Write file with index data, including garch volatility predictions
        self.pred_vols.to_csv('garch_index_vol.csv', index=False)

        # Write file with actual monthly volatility
        self.monthly_vol.to_csv('actual_monthly_vol.csv', index=False)