import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StocksMonthlyData:

    def __init__(self):
        '''
        Import data.
        '''

        # Get data
        self.comp_info = pd.read_csv("company_wrds.csv")

    
    def transform_comp_info(self):
        '''
        Apply initial tranformations to data coming from WRDS.
        '''

        # Calculate Market Cap as Price @ Close unadjusted x Shares outstanding
        self.comp_info['mkt_cap'] = self.comp_info.prccm * self.comp_info.cshom

        # Calculate Adjusted Closing Price
        self.comp_info['adj_price'] = self.comp_info.prccm / self.comp_info.ajexm

        # Convert the date column to datetime format
        self.comp_info['datadate'] = pd.to_datetime(self.comp_info['datadate'])

        # Convert the 'datadate' column to datetime format with every date having the first day of each month
        self.comp_info['datadate'] = pd.to_datetime(self.comp_info.datadate.dt.strftime('%Y-%m') + "-01")

        return self.comp_info
    

    def returns_per_group(self, group):
        '''
        Calculates returns for each stock.
        ---
        group: data referring to only one stock
        '''

        returns = {'ret': group['adj_price'].pct_change()}
        return pd.DataFrame(returns)


    def calculate_returns(self):
        '''
        Apply the returns_per_group function to the df.
        '''

        # Group the data by the 'cusip' (stock) column
        grouped = self.comp_info.groupby('cusip')

        # Apply the function to each group
        rets = self.returns_per_group(grouped)

        # Combine the original DataFrame 'comp_info' with the calculated returns
        self.comp_info = pd.concat([self.comp_info, rets], axis=1)

        return self.comp_info
    

    def lagged_returns(self, group):
        '''
        Gets 1-month lagged returns for each group.
        ---
        group: data referring to only one stock
        '''

        returns = {'ret_1m': group['ret'].shift(1)}
        return pd.DataFrame(returns)


    def get_lagged_returns(self):
        '''
        Apply the lagged_returns function to the df.
        '''

        # Group the data by the 'cusip' (stock) column
        grouped = self.comp_info.groupby('cusip')

        # Apply the function to each group
        rets_1m = self.lagged_returns(grouped)

        # Combine the original DataFrame 'comp_info' with the lagged returns
        self.comp_info = pd.concat([self.comp_info, rets_1m], axis=1)

        return self.comp_info


    ##################################################################################################################################

    ###############################################################################
    #                                                                             #
    #    Functions to apply Daniel-Moskowitz restrictions for stock selection     #
    #                                                                             #
    ###############################################################################


    def mkt_cap_filter(self, df, formation_month):
        '''
        Has Market Cap in formation month.
        ---
        df: dataframe to apply the function to
        formation_month: date corresponding to the formation month when stock is selected as part of the index
        '''

        applied_filter = df[df.datadate == formation_month].dropna(subset=['mkt_cap'])
        valid_cusips = applied_filter['cusip'].unique()

        return valid_cusips


    def rets_filter(self, df, valid_cusips, month_prior, formation_month):
        '''
        Has returns for formation date and month before.
        ---
        df: dataframe to apply the function to
        valid_cusips: list of valid company identifiers coming from previous filters
        month_prior: date corresponding to the month before the formation month
        formation_month: date corresponding to the formation month when stock is selected as part of the index
        '''
        
        # Look at the two months around the formation month and drop cusips where return is missing
        condition = ((df.cusip.isin(valid_cusips)) & (df.datadate >= month_prior) & (df.datadate <= formation_month))
        applied_filter = df[condition].dropna(subset=['ret'])
        
        # Keep only observations where a given cusip has returns for the two months
        val_counts = applied_filter['cusip'].value_counts()
        applied_filter = applied_filter[applied_filter['cusip'].isin(val_counts.index[val_counts == 2])]
        valid_cusips = applied_filter['cusip'].unique()

        return valid_cusips


    def price_filter(self, df, valid_cusips, year_prior):
        '''
        Has price last year.
        ---
        df: dataframe to apply the function to
        valid_cusips: list of valid company identifiers coming from previous filters
        year_prior: date corresponding to 1 year before the formation month
        '''
        
        condition = ((df.cusip.isin(valid_cusips)) & (df.datadate == year_prior))
        applied_filter = df[condition].dropna(subset=['adj_price'])
        valid_cusips = applied_filter['cusip'].unique()

        return valid_cusips


    def observations_filter(self, df, valid_cusips, year_prior, formation_month):
        '''
        Has 8 observations in the year before formation month (including).
        ---
        df: dataframe to apply the function to
        valid_cusips: list of valid company identifiers coming from previous filters
        year_prior: date corresponding to 1 year before the formation month
        formation_month: date corresponding to the formation month when stock is selected as part of the index                            
        '''

        condition = ((df.cusip.isin(valid_cusips)) & (df.datadate >= year_prior) & (df.datadate <= formation_month))
        applied_filter = df[condition]

        # Keep only observations where a given cusip is present at least 8 times
        val_counts = applied_filter['cusip'].value_counts()
        applied_filter = applied_filter[applied_filter['cusip'].isin(val_counts.index[val_counts >= 8])]

        # Keep only observations where of those 8 times, at least 4 have returns different to zero (to make sure stock is actually being traded)
        applied_filter = applied_filter[applied_filter['ret'] != 0]
        val_counts = applied_filter['cusip'].value_counts()
        applied_filter = applied_filter[applied_filter['cusip'].isin(val_counts.index[val_counts >= 4])]

        valid_cusips = applied_filter['cusip'].unique()

        return valid_cusips
    

    ##################################################################################################################################
    

    def get_data(self):
        '''
        Apply all functions to gather data.
        '''

        # Apply necessary functions to get comp_info correctly
        self.transform_comp_info()
        self.calculate_returns()
        self.get_lagged_returns()

        print("Getting data....")

        # Group the data by month and year
        grouped = self.comp_info.groupby(self.comp_info['datadate'])

        # Create list to store all data
        all_data_list = []

        # Iterate over each group (monthly), apply Moskowitz restrictions and select the top 600 stocks by mkt cap
        for name, group in grouped:

            print(f"Current Date: {name}")

            # Get date in the group
            formation_month = pd.to_datetime(group.iloc[0]['datadate'])
            
            # (Top 600 is collected in the last day of the month, and will only be used in the following)
            # Get the holding month date - date in the group + 1 month
            holding_month = formation_month + pd.DateOffset(months=1)

            # Get the date of year prior to formation month
            year_prior = formation_month - pd.DateOffset(months=12)

            # Get the date of month prior to formation month
            month_prior = formation_month - pd.DateOffset(months=1)

            # Apply Moskowitz restrictions

            # Market Cap filter
            valid_cusips = self.mkt_cap_filter(self.comp_info, formation_month)
            
            # Returns filter
            valid_cusips = self.rets_filter(self.comp_info, valid_cusips, month_prior, formation_month)
            
            # Price filter
            valid_cusips = self.price_filter(self.comp_info, valid_cusips, year_prior)

            # Observations filter
            valid_cusips = self.observations_filter(self.comp_info, valid_cusips, year_prior, formation_month)

            # Transform group to only keep observations that match the applied filters
            group = group[group.cusip.isin(valid_cusips)]

            # Get the top 600 by mkt cap
            top_600 = group.sort_values(by='mkt_cap', ascending=False).head(600)
            
            # Get the data for the holding month for the top 600
            condition = ((self.comp_info.cusip.isin(top_600.cusip.unique())) & (self.comp_info.datadate == holding_month))
            top600_data = self.comp_info[condition]

            # List of cusips that were part of formation month and are no longer in the data in the holding month
            missing_cusips = list(set(top_600.cusip.unique()) - set(top600_data.cusip.unique()))

            # Add to holding month missing data, assuming zero returns
            if not top600_data.empty: base = top600_data.iloc[0]
            new_data = []
            for cusip in missing_cusips:
                prev_obs = group[group.cusip == cusip]
                new_row = pd.DataFrame({'gvkey': prev_obs.gvkey, 'iid': prev_obs.iid, 'datadate': base.datadate, 'tic': prev_obs.tic, 'cusip': cusip,
                                        'conm': prev_obs.conm, 'ajexm': prev_obs.ajexm, 'prccm': prev_obs.prccm, 'cshom': prev_obs.cshom,
                                        'exchg': prev_obs.exchg, 'tpci': prev_obs.tpci, 'gsector': prev_obs.gsector, 'mkt_cap': prev_obs.mkt_cap,
                                        'adj_price': prev_obs.adj_price, 'ret': 0})
                new_data.append(new_row)
            if new_data:
                new_data = pd.concat(new_data, ignore_index=True)
                top600_data = pd.concat([top600_data, new_data])

            # Add to the holding month data
            all_data_list.append(top600_data)

        # Join all data collected into one df    
        if all_data_list:
            self.result_df = pd.concat(all_data_list)

        # Reset the index of the result DataFrames
        self.result_df.reset_index(drop=True, inplace=True)

        # Display the resulting DataFrame with the 600 largest market capitalizations per month
        return self.result_df
    

    def write_files(self):
        '''
        Extract dataframe to csv format.
        '''

        # Get the data
        self.get_data()

        # Write text file with gvkey's to extract from WRDS
        with open('gvkey.txt', 'w') as file:
            for gvkey in self.result_df.gvkey.unique():
                file.write(str(gvkey) + '\n')

        # Extract dataframe to csv format
        self.result_df.to_csv('index_components.csv', index=False)
        