import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import copy

class FinancialRatios:

    def __init__(self):
        '''
        Import all necessary data and do initial transformations.
        '''

        # Get index components data
        self.sec_month = pd.read_csv("index_components.csv")
        # Convert the date column to datetime format
        self.sec_month['datadate'] = pd.to_datetime(self.sec_month['datadate'])
        # Create formation month column
        self.sec_month['formation_month'] = self.sec_month['datadate'] - pd.DateOffset(months=1)

        # Get company ratios data
        self.com_ratios_gvkey = pd.read_csv("company_ratios_gvkey.csv")
        # Drop dates unnecessary, only want date when information is publicly available
        self.com_ratios_gvkey.drop(labels=['adate', 'qdate'], axis=1, inplace=True)
        # Rename columns
        self.com_ratios_gvkey.rename({'public_date': 'datadate', 'TICKER': 'tic'}, axis=1, inplace=True)
        # Convert the date column to datetime format
        self.com_ratios_gvkey['datadate'] = pd.to_datetime(self.com_ratios_gvkey['datadate'])
        # Convert the 'datadate' column to datetime format with every date having the first day of each month
        self.com_ratios_gvkey.datadate = self.com_ratios_gvkey.datadate.apply(lambda x: x.replace(day=1))

        # Get GICS codes dictionary
        self.gics_dict = pd.read_excel('gics_cods.xlsx').set_index('Code')['Sector'].to_dict()

        # Get industry ratios data
        self.ind_ratios = pd.read_csv("industry_ratios.csv")
        # Make date match used format
        self.ind_ratios.public_date = pd.to_datetime(self.ind_ratios.public_date)
        self.ind_ratios.public_date = self.ind_ratios.public_date.apply(lambda x: x.replace(day=1))


    def merge_com_ratios(self):
        '''
        Merge components data frame with company ratios.
        '''

        # Merge components data frame with company ratios on date where data is made available for the formation month
        self.merged_df = pd.merge(self.sec_month, self.com_ratios_gvkey, left_on=['formation_month','gvkey'], right_on=['datadate','gvkey'], how='inner')
        # Drop redundant columns
        self.merged_df.drop(["datadate_y", "tic_y", "cusip_y"], axis=1, inplace=True)
        # Rename columns
        self.merged_df.rename({"datadate_x":"datadate" , "tic_x": "tic", "cusip_x": "cusip"}, axis=1, inplace=True)


    def subset_com_ratios(self):
        '''
        Get top 500 by market cap.
        '''

        # Group the data by month and year
        grouped = self.merged_df.groupby(self.merged_df['datadate'])

        data = []

        # Iterate over each group (monthly) and select the top 500 stocks by mkt cap
        for _, group in grouped:

            # Get the top 500 by mkt cap for which we have data
            top_500 = group.sort_values(by='mkt_cap', ascending=False).head(500)

            # Add to the data
            data.append(top_500)

        # Join all data collected into one df    
        if data:
            self.sec_month = pd.concat(data)

        # Reset the index of the result DataFrames
        self.sec_month.reset_index(drop=True, inplace=True)

        # Drop unnecessary columns
        self.sec_month = self.sec_month.drop(['iid', 'ajexm', 'primiss', 'prccm', 'cshom', 'exchg', 'tpci', 'fic'], axis=1).sort_values(by=["datadate", "tic"])

    
    def transform_gics(self):
        '''
        Join sectors information.
        '''

        # Transform sector codes to names
        self.sec_month.gsector = self.sec_month.gsector.map(self.gics_dict)

        # Before August 31, 2016, the real estate sector was categorized as part of the financials sector within the GICS classification
        condition = (self.sec_month.gsector=="Real Estate") & (self.sec_month.datadate <= '2016-09-01')
        self.sec_month.loc[condition, 'gsector'] = "Financials"

    
    def add_difs(self, df, col_list):
        '''
        Create new columns with variations in different time-frames for all variables in col_list.
        ---
        df: dataframe to apply the function to
        col_list: list of columns for which to get variations
        '''

        for col in col_list:

            new_name_var1 = col + "_var1m"
            new_name_var3 = col + "_var3m"
            new_name_var6 = col + "_var6m"
            new_name_dif1 = col + "_dif1m"
            new_name_dif3 = col + "_dif3m"
            new_name_dif6 = col + "_dif6m"

            # Replace zeros by very small values, so as to not get infinite variations
            df_col_zeros = df[col].replace(0, 1e-3)

            df[new_name_var1] = ((df_col_zeros - df_col_zeros.shift(1)) / df_col_zeros.shift(1))
            df[new_name_var3] = ((df_col_zeros - df_col_zeros.shift(3)) / df_col_zeros.shift(3))
            df[new_name_var6] = ((df_col_zeros - df_col_zeros.shift(6)) / df_col_zeros.shift(6))
            df[new_name_dif1] = df[col] - df[col].shift(1)
            df[new_name_dif3] = df[col] - df[col].shift(3)
            df[new_name_dif6] = df[col] - df[col].shift(6)
        
        return df
    

    def fill_sector_missing(self, df, sector_col='gicdesc', missing_sector='Real Estate', ratio_cols=None, base_sector='Financials'):
        '''
        Fill 'Real Estate' sector variations data with 'Financials' variations data where it is missing.
        ---
        df: dataframe to apply the function to
        sector_col: column that contains the information on which sector the firm belongs to
        missing_sector: sector for which we will replace missing information
        ratio_cols: columns with ratios/variations for which we will replace missing variations
        base_sector: sector where information to replace missing will be collected
        '''

        # Filter rows for given sector where there are missing values
        missing = df[(df[sector_col] == missing_sector) & df[ratio_cols].isnull().any(axis=1)]

        # Iterate through each row in the filtered dataframe
        for index, row in missing.iterrows():
            
            date = row['public_date']

            # Find the corresponding row in base sector for the same date
            financials_row = df[(df['public_date'] == date) & (df[sector_col] == base_sector)].iloc[0]

            # Replace missing values
            df.loc[index, ratio_cols] = financials_row[ratio_cols]

        return df


    def add_difs_ind(self):
        '''
        Apply both the add_difs and the fill_sector_missing functions to the df.
        '''

        # Group by sector
        groups = self.ind_ratios.groupby('gicdesc')

        # Add difs for each industry financial ratio column
        data=[]
        for _, group in groups:

            self.add_difs(group, col_list=['bm_Median', 'CAPEI_Median', 'divyield_Median', 'pe_exi_Median', 'ptb_Median',
                                            'roa_Median', 'roe_Median', 'short_debt_Median', 'curr_debt_Median',
                                            'debt_ebitda_Median', 'debt_assets_Median', 'cash_conversion_Median',
                                            'cash_ratio_Median', 'curr_ratio_Median', 'quick_ratio_Median'])
            
            data.append(group)

        if data: self.ind_ratios = pd.concat(data)

        # Re-sort index
        self.ind_ratios = self.ind_ratios.sort_index()

        # Fill missing data for Real Estate variations
        var_cols =  ['bm_Median_var1m', 'bm_Median_var3m', 'bm_Median_var6m', 'CAPEI_Median_var1m', 'CAPEI_Median_var3m', 'CAPEI_Median_var6m',
                    'divyield_Median_var1m', 'divyield_Median_var3m', 'divyield_Median_var6m', 'pe_exi_Median_var1m', 'pe_exi_Median_var3m', 'pe_exi_Median_var6m',
                    'ptb_Median_var1m', 'ptb_Median_var3m', 'ptb_Median_var6m', 'roa_Median_var1m', 'roa_Median_var3m', 'roa_Median_var6m',
                    'roe_Median_var1m', 'roe_Median_var3m', 'roe_Median_var6m', 'short_debt_Median_var1m', 'short_debt_Median_var3m', 'short_debt_Median_var6m',
                    'curr_debt_Median_var1m', 'curr_debt_Median_var3m', 'curr_debt_Median_var6m', 'debt_ebitda_Median_var1m', 'debt_ebitda_Median_var3m', 'debt_ebitda_Median_var6m',
                    'debt_assets_Median_var1m', 'debt_assets_Median_var3m', 'debt_assets_Median_var6m', 'cash_conversion_Median_var1m', 'cash_conversion_Median_var3m', 'cash_conversion_Median_var6m',
                    'cash_ratio_Median_var1m', 'cash_ratio_Median_var3m', 'cash_ratio_Median_var6m', 'curr_ratio_Median_var1m', 'curr_ratio_Median_var3m', 'curr_ratio_Median_var6m',
                    'quick_ratio_Median_var1m', 'quick_ratio_Median_var3m', 'quick_ratio_Median_var6m']
        
        self.fill_sector_missing(self.ind_ratios, ratio_cols=var_cols)


    def merge_ind_ratios(self):
        '''
        Merge industry ratios.
        '''

        # Merge data frame with industry ratios on date where data is made available for the formation month
        self.sec_month = self.sec_month.merge(self.ind_ratios, left_on=['formation_month', 'gsector'], right_on=['public_date', 'gicdesc'], how='left')
        # Drop redundant columns
        self.sec_month.drop(['public_date', 'gicdesc'], axis=1, inplace=True)


    def form_vw_vars(self):
        '''
        Get Value-Weighted Index related columns.
        '''

        # Calculate the total market cap for each day
        self.sec_month['total_market_cap'] = self.sec_month.groupby('datadate')['mkt_cap'].transform('sum')

        # Calculate the weight of each company's market cap
        self.sec_month['weight'] = self.sec_month['mkt_cap'] / self.sec_month['total_market_cap']

        # Calculate the weighted return for each company
        self.sec_month['weighted_return'] = self.sec_month['ret'] * self.sec_month['weight']

        # Group by date and calculate the sum of weighted returns for each day
        grouped_data = self.sec_month.groupby('datadate')['weighted_return'].sum().reset_index()        

        self.sec_month = pd.merge(self.sec_month, grouped_data, on="datadate")
        self.sec_month.rename({"weighted_return_x":"weighted_return", "weighted_return_y":"vw_return"}, axis=1, inplace=True)


    def lag_index_returns(self):
        '''
        Get lagged index returns.
        '''

        index_returns = self.sec_month.rename({'vw_return': 'vw_return_1m'}, axis=1).groupby('datadate')['vw_return_1m'].first().shift(1)
        self.sec_month = pd.merge(self.sec_month, index_returns, on="datadate")

    
    def lag_weights(self):
        '''
        Get lagged Value-Weighted weights.
        '''
        
        weights_1m = self.sec_month[['datadate', 'gvkey', 'weight']].rename({'weight':'weight_1m'}, axis=1)
        weights_1m.datadate = weights_1m.datadate - pd.DateOffset(months=1)

        self.sec_month = pd.merge(self.sec_month, weights_1m, on=['datadate', 'gvkey'])

        # Keep only dates for which we have industry ratios
        self.financial_ratios = copy.deepcopy(self.sec_month[self.sec_month.datadate >= "2000-01-01"])

    
    def replace_median(self, df, column):
        '''
        Replace missing firm ratios with corresponding industry median.
        ---
        df: dataframe to apply the function to
        column: column for which missing values will be replaced
        '''

        # Where company ratios aren't in the data, replace by industry median
        df.loc[df[column].isna(), column] = df.loc[df[column].isna(), f"{column}_Median"] 

    
    def apply_replace_median(self, column_list=['CAPEI', 'bm', 'pe_exi', 'roa', 'roe', 'debt_ebitda', 'debt_assets', 'ptb']):
        '''
        Apply replace median function to df.
        ---
        column_list: list of columns for which missing values will be replaced
        '''

        # Replace NA values in each column by the corresponding industry median
        for column in column_list: self.replace_median(self.financial_ratios, column)

    
    def compare_industry(self, df, column):
        '''
        Get comparison between firm ratio and industry ratio.
        ---
        df: dataframe to apply the function to
        column: column that has firm ratio for which we will get the comparison
        '''

        # Replace zeros by very small values, so as to not get infinite returns
        col_zeros = df[column].replace(0, 1e-3)
        median_zeros = df[f"{column}_Median"].replace(0, 1e-3)

        # Create new column that compares the firm's financial ratio with the indsutry median
        df[f"{column}_comp"] = col_zeros / median_zeros

    
    def apply_compare_industry(self, column_list=['CAPEI', 'bm', 'pe_exi', 'roa', 'roe', 'debt_ebitda', 'debt_assets', 'ptb']):
        '''
        Apply compare industry function to df.
        ---
        df: dataframe to apply the function to
        column_list: list of column (ratios) to get the comparison
        '''
                
        for column in column_list: self.compare_industry(self.financial_ratios, column) 


    def get_index_components(self):
        '''
        Get new dataframe with only information on which are the index components.
        '''

        # Reduce the sec month df, to keep only relevant columns
        self.index_components = self.sec_month[["gvkey", "datadate", "tic"]]


    def get_data(self):
        '''
        Apply all functions to get the desired df.
        '''

        self.merge_com_ratios()
        self.subset_com_ratios()
        self.transform_gics()
        self.add_difs_ind()
        self.merge_ind_ratios()
        self.form_vw_vars()
        self.lag_index_returns()
        self.lag_weights()
        self.apply_replace_median()
        self.apply_compare_industry()
        self.get_index_components()

        return self.financial_ratios
    

    def write_files(self):
        '''
        Extract dataframes to csv format.
        '''

        # Get the data
        self.get_data()

        # Write files with gvkey's to extract from WRDS
        with open('gvkey_daily.txt', 'w') as file:
            for gvkey in self.sec_month.gvkey.unique():
                file.write(str(gvkey) + '\n')

        # Extract financial ratios dataframe to csv format
        self.financial_ratios.to_csv('financial_ratios.csv', index=False)

        # Extract index components dataframe to csv format (write over previous file)
        self.index_components.to_csv('index_components.csv', index=False)


            

        




    







