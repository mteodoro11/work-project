# Import packages
from fredapi import Fred
import pandas as pd

class MacroData:

    def __init__(self):

        # Get API
        self.fred = Fred(api_key='4affddb93a7983c159d5af932c33cf38')


    def extract_initial_claims(self):
        '''
        Transform initial claims data into monthly average.
        '''

        self.init_claims = self.fred.get_series("ICSA")
        self.init_claims = self.init_claims.groupby(self.init_claims.index.to_period('M')).mean()
        self.init_claims.index = self.init_claims.index.strftime('%Y-%m-01')
        self.init_claims.index = pd.to_datetime(self.init_claims.index)
        
        return self.init_claims


    def extract_gold(self):
        '''
        Get gold data obtained from WGC.
        '''

        self.gold = pd.read_excel('gold_eop.xlsx')
        self.gold = self.gold.set_index('Date')['Gold']
        self.gold.index = self.gold.index.strftime('%Y-%m-01')
        self.gold.index = pd.to_datetime(self.gold.index)
        
        return self.gold


    def add_difs(self, df, col_list):
        '''
        Create new columns with variations in different time-frames for all variables in col_list.
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


    def extract_data(self, start_date):
        '''
        Form dataframe from all gathered information.
        '''

        self.init_claims = self.extract_initial_claims()
        self.gold = self.extract_gold()

        self.fred_data = pd.DataFrame({'Fed_Funds_Rate': self.fred.get_series("FEDFUNDS", observation_start=start_date), # Fed Funds Rate
                'Unemployment_Rate': self.fred.get_series("UNRATE", observation_start=start_date),                  # Unemployment Rate
                'GDP_Growth_Rate': self.fred.get_series("A191RP1Q027SBEA", observation_start=start_date),           # GDP Growth Rate
                'Debt_to_GDP': self.fred.get_series("GFDEGDQ188S", observation_start=start_date),                   # Debt as a % of GDP
                'CPI': self.fred.get_series("CPIAUCSL", observation_start=start_date),                              # CPI
                'PCE': self.fred.get_series("PCEPI", observation_start=start_date),                                 # Personal Consumption Expenditures
                'Core_PCE': self.fred.get_series("PCEPILFE", observation_start=start_date),                         # Personal Consumption Expenditures Excluding Food and Energy
                'EUR/USD': self.fred.get_series("EXUSEU", observation_start=start_date),                            # EUR/USD Spot Rate
                'GBP/USD': self.fred.get_series("EXUSUK", observation_start=start_date),                            # GBP/USD Spot Rate
                'USD/JPY': self.fred.get_series("EXJPUS", observation_start=start_date),                            # USD/JPY Spot Rate
                'Crude_Oil_WTI': self.fred.get_series("MCOILWTICO", observation_start=start_date),                  # Crude Oil Prices: West Texas Intermediate (WTI)
                'Crude_Oil_Brent': self.fred.get_series("MCOILBRENTEU", observation_start=start_date),              # Crude Oil Prices: Brent - Europe
                'Gold': self.gold[self.gold.index >= start_date],                                                   # Gold Spot Price
                'Copper': self.fred.get_series("PCOPPUSDM", observation_start=start_date),                          # Copper Spot Price
                'Wheat': self.fred.get_series("PWHEAMTUSDM", observation_start=start_date),                         # Wheat Spot Price
                '3m_yield': self.fred.get_series("GS3M", observation_start=start_date),                             # Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity
                '1y_yield': self.fred.get_series("GS1", observation_start=start_date),                              # Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity
                '2y_yield': self.fred.get_series("GS2", observation_start=start_date),                              # Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity
                '5y_yield': self.fred.get_series("GS5", observation_start=start_date),                              # Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity
                '10y_yield': self.fred.get_series("GS10", observation_start=start_date),                            # Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
                '20y_yield': self.fred.get_series("GS20", observation_start=start_date),                            # Market Yield on U.S. Treasury Securities at 20-Year Constant Maturity
                '30y_yield': self.fred.get_series("GS30", observation_start=start_date),                            # Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity
                '10y2y_yield': self.fred.get_series("T10Y2YM", observation_start=start_date),                       # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
                '10y3m_yield': self.fred.get_series("T10Y3MM", observation_start=start_date),                       # 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
                'Initial_Claims': self.init_claims[self.init_claims.index >= start_date],                           # Initial Jobless Claims
                })
        
        # Forward fill missing values in GDP data
        self.fred_data.GDP_Growth_Rate = self.fred_data.GDP_Growth_Rate.ffill()
        self.fred_data.Debt_to_GDP = self.fred_data.Debt_to_GDP.ffill()

        # Drop the last row
        self.fred_data = self.fred_data.drop(self.fred_data.index[-1])


    def apply_add_difs(self):
        '''
        Apply the add_diffs function to the df.
        '''

        # Gather column list to apply difs function
        cols = list(self.fred_data.columns)
        cols.remove('GDP_Growth_Rate')

        self.add_difs(self.fred_data, cols)

        # Drop Debt-to-GDP 1-month variation as it doesnt make sense (3-month variation serves the same purpose)
        self.fred_data.drop("Debt_to_GDP_var1m", axis=1, inplace=True)

        self.fred_data = self.fred_data.reset_index().rename({'index':'datadate'}, axis=1)

        return self.fred_data
    

    def shift_cols(self, gdp_cols=["GDP_Growth_Rate", "Debt_to_GDP"]):
        '''
        Shift columns so that data always matches month when available (to not commit look-ahead bias).
        '''

        for col in self.fred_data.columns:

            # If not GDP data, available one month after
            if col not in gdp_cols and col != 'datadate':
                self.fred_data[col] = self.fred_data[col].shift(1)

            # If GDP, has to be 6 month lag, due to use of final data
            elif col in gdp_cols and col != 'datadate':
                self.fred_data[col] = self.fred_data[col].shift(6)
        
        # Correct df dates
        self.fred_data = self.fred_data[((self.fred_data.datadate >= "2000-01-01") & (self.fred_data.datadate < "2023-01-01"))]
    

    def get_data(self, start_date):
        '''
        Apply all functions to get df.
        '''

        self.extract_data(start_date)
        self.apply_add_difs()
        self.shift_cols()


    def macro_to_csv(self, start_date):
        '''
        Extract dataframe to csv format.
        '''

        self.get_data(start_date)
        self.fred_data.to_csv('fred_data.csv', index=False)
