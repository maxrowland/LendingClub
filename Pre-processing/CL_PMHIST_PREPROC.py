#Author: Max H. Rowland
#Email: maxh.rowland@gmail.com
#Script cleans LendingClub payment history files for loading into MySQL server
#%% Import packages
import pandas as pd
import numpy as np
import time
pd.options.mode.use_inf_as_na = True #sets infinite values to na
#%% Select which files you want to preprocess
run_pmt_all = True
run_pmt_all_status = False
#%%Clean PMTHIST_ALL data file for upload into MySQL server
if run_pmt_all is True:
    start_time = time.time()
    pmt_all = pd.read_csv('E:\DATABASES\LENDING_CLUB\PAYMENT_HIST_DATA\ORIGINAL\PMTHIST_ALL_202004.csv')

    #Convert MONTH to excel short date
    pmt_all['MONTH'] = pmt_all['MONTH'].astype('datetime64[ns]')

    #Drop columns - duplicate information
    pmt_all.drop('InterestRate', axis=1, inplace=True)
    pmt_all.drop('IssuedDate', axis=1, inplace=True)
    pmt_all.drop('MONTHLYCONTRACTAMT', axis=1, inplace=True)
    pmt_all.drop('dti', axis=1, inplace=True)
    pmt_all.drop('State', axis=1, inplace=True)
    pmt_all.drop('HomeOwnership', axis=1, inplace=True)
    pmt_all.drop('EarliestCREDITLine', axis=1, inplace=True)
    pmt_all.drop('OpenCREDITLines', axis=1, inplace=True)
    pmt_all.drop('TotalCREDITLines', axis=1, inplace=True)
    pmt_all.drop('RevolvingCREDITBalance', axis=1, inplace=True)
    pmt_all.drop('RevolvingLineUtilization', axis=1, inplace=True)
    pmt_all.drop('Inquiries6M', axis=1, inplace=True)
    pmt_all.drop('DQ2yrs', axis=1, inplace=True)
    pmt_all.drop('MonthsSinceDQ', axis=1, inplace=True)
    pmt_all.drop('PublicRec', axis=1, inplace=True)
    pmt_all.drop('MonthsSinceLastRec', axis=1, inplace=True)
    pmt_all.drop('EmploymentLength', axis=1, inplace=True)
    pmt_all.drop('currentpolicy', axis=1, inplace=True)
    pmt_all.drop('grade', axis=1, inplace=True)
    pmt_all.drop('APPL_FICO_BAND', axis=1, inplace=True)
    pmt_all.drop('MonthlyIncome', axis=1, inplace=True)
    pmt_all.drop('RECEIVED_D', axis=1, inplace=True)
    pmt_all.drop('Last_FICO_BAND', axis=1, inplace=True)
    pmt_all.drop('CO', axis=1, inplace=True)

    #Pre Payment Status
    pmt_all['PRE_PAY'] = np.where((pmt_all['PERIOD_END_LSTAT'] == 'Fully Paid')&(pmt_all['MOB']<pmt_all['term']),1,0)
    pmt_all.drop('term', axis=1, inplace=True)

    #Charge Off Starus
    pmt_all['CHARGE_OFF'] = np.where((pmt_all['PERIOD_END_LSTAT'] == 'Charged Off'),1,0)

    #Monthly net return calculations
    pmt_all['LC_FEE'] = (.01 * pmt_all['RECEIVED_AMT']).where(pmt_all['MOB'] > 12, (.01 * pmt_all['DUE_AMT']))
    pmt_all['NET_RECOVERY'] = pmt_all['PCO_RECOVERY'] - pmt_all['PCO_COLLECTION_FEE']
    pmt_all['NET_RECOVERY'] = pmt_all['NET_RECOVERY'].where(pmt_all['NET_RECOVERY'].isnull(),pmt_all['NET_RECOVERY']).fillna(0).astype(float)
    pmt_all['RESIDUAL_RECEIVED'] = pmt_all['RECEIVED_AMT'] - pmt_all['INT_PAID'] - pmt_all['PRNCP_PAID'] - pmt_all['FEE_PAID']
    pmt_all['NET_GL'] = pmt_all['INT_PAID'] + pmt_all['FEE_PAID'] + pmt_all['NET_RECOVERY'] + pmt_all['RESIDUAL_RECEIVED'] - pmt_all['LC_FEE'] - pmt_all['COAMT']
    pmt_all['NET_GL_CUM'] = pmt_all.groupby('LOAN_ID')['NET_GL'].cumsum()
    pmt_all['BV'] = pmt_all['PBAL_BEG_PERIOD'].where(pmt_all['MOB'] == 1, (pmt_all.groupby('LOAN_ID')['PBAL_BEG_PERIOD'].transform('first') + pmt_all['NET_GL_CUM'].shift(1)))
    pmt_all['EV'] = pmt_all['BV'] + pmt_all['NET_GL']
    pmt_all['NET_RETURN'] = pmt_all['EV'] / pmt_all['BV'] - 1

    #Convert MONTH to excel short date
    pmt_all['MONTH'] = pmt_all['MONTH'].astype('datetime64[ns]').dt.strftime('%m/%d/%Y')

    #Export to CSV
    pmt_all.to_csv('E:\DATABASES\LENDING_CLUB\PAYMENT_HIST_DATA\PREPROCESSED\PMTHIST_ALL_202004.csv',index=False)
    print(print("--- %s seconds ---" % (time.time() - start_time)))

#%%Slices out Loan status and place into DataFrame then saves to csv
if run_pmt_all_status is True:
    new_loan_status = pmt_all[['LOAN_ID','MONTH','MOB','PERIOD_END_LSTAT']]
    new_loan_status = pmt_all.groupby('LOAN_ID')['LOAN_ID','MONTH','PERIOD_END_LSTAT', 'PRE_PAY'].transform('last').drop_duplicates(keep='last')

    new_loan_status.to_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\new_loan_status.csv', index=False)
    print(new_loan_status['PERIOD_END_LSTAT'].value_counts(dropna=False),new_loan_status['PERIOD_END_LSTAT'].value_counts(normalize=True, dropna=False))
    print(print("--- %s seconds ---" % (time.time() - start_time)))
