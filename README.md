# work-project
 Work Project Master's in Finance Thesis at Nova SBE 22/23

Data files containd data retrieved from S&P Global Market Intelligence Data Through Wharton Research Data Services (WRDS) and therefore are not allowed to be publicly shared, nevertheless, results of the developed analysis can be found in this repository.

Data gathering stucture is constructed as follows:

![image](https://github.com/mteodoro11/work-project/assets/78867736/9e1f1df9-3ec5-4296-adcb-bf62886d4af4)

The following scripts gather the required data. The data-gatherer.ipynb notebook calls all scripts and joins all data collected into a single data frame.

- fred_extract.py
- index_components_extract.py
- financial_ratios.py
- daily_extract.py
- market_betas.py
- build_final.py

Using the created data frame, we check for the validity of assumptions from Moreira and Muir (2017) at the stock-level in the assumptions.ipynb notebook.

Model construction for stock volatility forecasting and consequent evaluation are presented in the modelling.ipynb notebook.

Finally, portfolio construction and assessment based on the forecasted volatilities is developed in the portfolio-assessment.ipynb notebook.
