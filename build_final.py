import pandas as pd

class FinalData:

    def __init__(self):

        # Import files
        self.financial_ratios = pd.read_csv("financial_ratios.csv")
        self.fred = pd.read_csv("fred_data.csv")
        self.garch_index = pd.read_csv("garch_index_vol.csv")
        self.monthly_vol = pd.read_csv("actual_monthly_vol.csv")
        self.prev_vol = pd.read_csv("vol_previous_month.csv")
        self.market_betas = pd.read_csv("market_betas.csv")
        self.rf_rate = pd.read_csv("ff_factors.csv")

        # Rename necessary columns
        self.monthly_vol.rename(columns={'datadate_x': 'datadate'}, inplace=True)
        self.garch_index.rename({'volatility_predicted':'garch_vol', 'volatility_actual':'index_volatility', 'volatility_1m':'index_volatility_1m'}, axis=1, inplace=True)
        self.market_betas.rename({'yearmonth':'datadate'}, axis=1, inplace=True)

        # Subset dates to get only the wanted set
        self.fred = self.fred[((self.fred.datadate >= "2000-01-01") & (self.fred.datadate < "2023-01-01"))]
        self.prev_vol = self.prev_vol[((self.prev_vol.datadate >= "2000-01-01") & (self.prev_vol.datadate < "2023-01-01"))]
        self.monthly_vol = self.monthly_vol[((self.monthly_vol.datadate >= "2000-01-01") & (self.monthly_vol.datadate < "2023-01-01"))]


    def fix_date_types(self):

        # Fix date types
        self.financial_ratios.datadate = pd.to_datetime(self.financial_ratios.datadate)
        self.fred.datadate = pd.to_datetime(self.fred.datadate)
        self.garch_index.datadate = pd.to_datetime(self.garch_index.datadate + "-01")
        self.monthly_vol.datadate = pd.to_datetime(self.monthly_vol.datadate)
        self.prev_vol.datadate = pd.to_datetime(self.prev_vol.datadate)
        self.market_betas.datadate = pd.to_datetime(self.market_betas.datadate + '-01')
    
    def merge_betas(self):

        self.financial_ratios = pd.merge(self.financial_ratios, self.market_betas, on=["datadate", "gvkey"], how="left")
        self.financial_ratios.market_beta = self.financial_ratios.market_beta.fillna(0)

    
    def merge_fred(self):

        self.df = self.financial_ratios.merge(self.fred)


    def merge_monthly_vol(self):

        self.df = self.df.merge(self.monthly_vol, on=['gvkey', 'datadate'], how='left')

        # Replace missing volatitilities with 7%
        self.df.loc[self.df.vol.isna(), 'vol'] = 0.07

    
    def merge_prev_vol(self):

        self.df = self.df.merge(self.prev_vol, on=['gvkey','datadate'], how='left')

        # Replace missing volatitilities with 7%
        self.df.loc[self.df.vol_1m.isna(), 'vol_1m'] = 0.07

    
    def merge_garch_index(self):

        self.df = self.df.merge(self.garch_index, on='datadate')

    
    def add_rf_feats(self):

        self.rf_rate = self.rf_rate[['Date', 'RF']]
        self.rf_rate.RF = self.rf_rate.RF.shift(1)
        self.rf_rate.loc[:,"RF"] = self.rf_rate.loc[:,"RF"] / 100
        self.rf_rate = self.rf_rate[(self.rf_rate.Date >= "2000-01") & (self.rf_rate.Date < "2023-01")].reset_index(drop="True")
        self.rf_rate.rename({"Date":"datadate", "RF":"rf_1m"}, axis=1, inplace=True)
        self.rf_rate.datadate = pd.to_datetime(self.rf_rate.datadate + "-01")

        self.df = self.df.merge(self.rf_rate, on='datadate')
        self.df['sr_1m'] = (self.df['ret_1m'] - self.df['rf_1m']) / self.df['vol_1m']
        self.df['index_sr_1m'] = (self.df['vw_return_1m'] - self.df['rf_1m']) / self.df['index_volatility_1m']


    def get_data(self):

        self.fix_date_types()
        self.merge_betas()
        self.merge_fred()
        self.merge_monthly_vol()
        self.merge_prev_vol()
        self.merge_garch_index()
        self.add_rf_feats()

        return self.df
    

    def write_file(self):

        self.get_data()

        # Write file with final df
        self.df.to_csv("wp_df.csv", index=False)


    