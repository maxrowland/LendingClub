#Author: Max H. Rowland
#Email: maxh.rowland@gmail.com
#Script pre-processes LendingClub data for use in a classifier model
#Import Packages
import numpy as np
import os, errno
import smtplib, ssl
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
pd.options.mode.use_inf_as_na = True #sets infinite values to na
pd.set_option('display.max_colwidth', 3000, 'display.max_rows', None, 'display.max_columns', None)
#Global variables & save folder creation
show_charts = False #set to True to show and save all plots
mblue = "#008AC9" #chart colors
mpurple = "#2B115A" #chart colors
mred = "#f11a21" #chart colors
color = 'black' #chart colors
save_path = r'E:\DATABASES\LENDING_CLUB\MODEL DATA\PREPROC_OUTPUT' #Directory for saving pre-processing output
timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S') #Data output save folder creation
start_time = time.time() #start time for execution timer

#Creates save folder name with date
def filecreation(list, filename):
    mydir = os.path.join(
        save_path,
        timestamp)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    with open(os.path.join(mydir, filename), 'w') as d:
        d.writelines(list)
filecreation(save_path, 'xxx')
folder_path = save_path + "\\" + timestamp

#%%Connect to MySQL Database and load data into pandas DataFrame + load feature descriptions and new loan status files
try:
    cnx1 = mysql.connector.connect(host='XXXX',
                                   port=XXXX,
                                   database='XXXX',
                                   user='XXXX',
                                   password='XXXX')

#Query cnx1 MySQL Databases
    sql_select_query = 'SELECT * from lendingclub.loans_all'
    cur1 = cnx1.cursor()
    cur1.execute('SELECT * FROM loans_all')
    loans = cur1.fetchall()

#Create DataFrame & apply column names
    loans = pd.DataFrame(loans, columns=cur1.column_names)

#Closes MySQL Connection or Displays Connection Error
except Error as e:
    print('Error reading data from MySQL table', e)
finally:
    if cnx1.is_connected():
        cnx1.close()
        cur1.close()
        print('MySQL Connection is Closed')
        print('--- MySQL Data Load: %s seconds ---' % (time.time() - start_time))

#Load feature description file
feat_desc = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\feature_descriptions.CSV')

#load updated new_loan_status
new_loan_status = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\new_loan_status.CSV')

#Loan Treasury Historical Yield Curve
mkt_yield = pd.read_excel(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\ECON_DATA.XLSX', sheet_name='YIELDS', index_col=0, skiprows=2)
#%%Replace loan_status column with new_loan_status data
loans['id'] = loans['id'].apply(int)
new_loan_status.drop('MONTH', axis=1, inplace=True)
new_loan_status['LOAN_ID'] = new_loan_status['LOAN_ID'].apply(int)
new_loan_status.rename(columns={'LOAN_ID':'id','PERIOD_END_LSTAT':'new_loan_status'}, inplace=True)
loans = pd.merge(loans, new_loan_status, left_on='id', right_on='id')
loans.drop('loan_status', axis=1, inplace=True)
loans = loans.set_index('id')

#%%Filter out loans that are not either Fully Paid or Charged Off and reclassify those categories that should be Fully paid of charged off
print("----------ORIGINAL LOANS SHAPE----------",file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(loans.shape,file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print("----------ORIGNAL LOAN STATUS DISTRIBUTION----------",file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print((loans['new_loan_status'].value_counts(dropna=False),loans['new_loan_status'].value_counts(normalize=True, dropna=False)),file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
loans['new_loan_status'].replace(to_replace='Does not meet the credit policy. Status:Fully Paid', value='Fully Paid', inplace=True)
loans['new_loan_status'].replace(to_replace='Does not meet the credit policy. Status:Charged Off', value='Charged Off', inplace=True)
loans['new_loan_status'].replace(to_replace='Default', value='Charged Off', inplace=True)
loans = loans.loc[loans['new_loan_status'].isin(['Fully Paid', 'Charged Off',])]
print(loans['new_loan_status'].value_counts(dropna=False))
print("----------LOAN STATUS DISTRIBUTION POST RECLASSIFICATION----------",file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print((loans['new_loan_status'].value_counts(dropna=False),loans['new_loan_status'].value_counts(normalize=True, dropna=False)),file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))

#%%Pre-Process some features before charting and exploratory analysis
#Convert date features to datetime
loans['issue_d'] = pd.to_datetime(loans['issue_d'],format='%m/%d/%Y')
loans['earliest_cr_line'] = pd.to_datetime(loans['earliest_cr_line'],format='%m/%d/%Y')

#convert nulls to Max
loans['mths_since_last_major_derog'] = loans['mths_since_last_major_derog'].fillna(loans['mths_since_last_major_derog'].max())
loans['mths_since_recent_bc_dlq'] = loans['mths_since_recent_bc_dlq'].fillna(loans['mths_since_recent_bc_dlq'].max())
loans['mths_since_last_delinq'] = loans['mths_since_last_delinq'].fillna(loans['mths_since_last_delinq'].max())
loans['mths_since_last_record'] = loans['mths_since_last_record'].fillna(loans['mths_since_last_record'].max())
loans['mths_since_recent_revol_delinq'] = loans['mths_since_recent_revol_delinq'].fillna(loans['mths_since_recent_revol_delinq'].max())
loans['mths_since_rcnt_il'] = loans['mths_since_rcnt_il'].fillna(loans['mths_since_rcnt_il'].max())
loans['mths_since_recent_inq'] = loans['mths_since_recent_inq'].fillna(loans['mths_since_recent_inq'].max())

#Convert nulls to 0
loans['annual_inc_joint'] = loans['annual_inc_joint'].fillna(0)
loans['dti_joint'] = loans['dti_joint'].fillna(0)
loans['il_util'] = loans['il_util'].fillna(0)
loans['total_bal_il'] = loans['total_bal_il'].fillna(0)
loans['tot_coll_amt'] = loans['tot_coll_amt'].fillna(0)
loans['tot_cur_bal'] = loans['tot_cur_bal'].fillna(0)

#%%Drop features that either are not of value, not available for new loans, or missing more than 30% of their data
missing_frac = loans.isnull().mean().sort_values(ascending=False)
print("----------FEATURE MISSING FRACTIONS----------",file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(missing_frac,file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
#Drop features list, if no comment these are features that are not available for new loans
drop_list = [
            'funded_amnt_inv',
            'pymnt_plan',
            'out_prncp',
            'out_prncp_inv',
            'total_pymnt',
            'total_pymnt_inv',
            'total_rec_prncp',
            'total_rec_int',
            'total_rec_late_fee',
            'recoveries',
            'collection_recovery_fee',
            'last_pymnt_d',
            'last_pymnt_amnt',
            'next_pymnt_d',
            'last_credit_pull_d',
            'last_fico_range_high',
            'last_fico_range_low',
            'policy_code',
            'hardship_flag',
            'hardship_type',
            'hardship_reason',
            'hardship_status',
            'deferral_term',
            'hardship_amount',
            'hardship_start_date',
            'hardship_end_date',
            'payment_plan_start_date',
            'hardship_length',
            'hardship_dpd',
            'hardship_loan_status',
            'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount',
            'hardship_last_payment_amount',
            'debt_settlement_flag',
            'debt_settlement_flag_date',
            'settlement_status',
            'settlement_date',
            'settlement_amount',
            'settlement_percentage',
            'settlement_term',
            'url', #no predictive power, a url to the loan listing
            'desc', #missing data after 2013
            'title', #same data as purpose column
            'emp_title', #unstructured data - look into nlp to map to BLS occupational handbook data at a future date
            'funded_amnt', #duplicate of loan amount
            'grade' #overlap with subgrade
            ]
print("----------FEATURES DROPPED BY USER----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(drop_list, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
loans.drop(labels=drop_list, axis=1, inplace=True)

#%%Drop features that are missing more than 50% of their data
missing_frac_post_drop = loans.isnull().mean().sort_values(ascending=False)
missing_drop = sorted(list(missing_frac_post_drop[missing_frac_post_drop > 0.10].index))
loans.drop(labels=missing_drop, axis=1, inplace=True)
print("----------FEATURES DROPPED WITH >10% MISSING DATA----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(missing_drop, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print("----------LOANS SHAPE: POST DROP----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(loans.shape,file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
del missing_frac_post_drop, missing_drop, drop_list

#%%Examine each feature workbench
def plot_var(col_name, full_name, continuous):
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), dpi=200)
    # Plot without loan status
    if continuous:
        sns.distplot(loans.loc[loans[col_name].notnull(), col_name], color=mpurple, kde=False, kde_kws={"color": mpurple}, ax=ax1, hist=True)
    else:
        sns.countplot(loans[col_name], order=sorted(loans[col_name].unique()), color=mpurple, saturation=1, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('Count')
    ax1.set_title(full_name, fontsize= 20)
    plt.savefig(folder_path + "\\" + col_name, dpi=200)
    # Plot with loan status
    if continuous:
        my_pal = {"Fully Paid": mblue, "Charged Off": mred}
        sns.boxplot(x=col_name, y='new_loan_status', palette=my_pal, data=loans, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status', fontsize= 20)
    else:
        charge_off_rates = loans.groupby(col_name)['new_loan_status'].value_counts(normalize=True).loc[:, 'Charged Off']
        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color=mblue, saturation=1, ax=ax2)
        ax2.set_ylabel('Percentage of Loans Charged off')
        ax2.set_title('Charge off Rate by ' + full_name, fontsize=20)
    ax2.set_xlabel('')
    plt.figtext(0.99, 0.01, 'Max H. Rowland', color=mred, horizontalalignment='right')
    plt.figtext(0.01, 0.03, 'Observations: '"{:,}".format(len(loans[col_name])), color=color,horizontalalignment='left')
    plt.figtext(0.01, 0.01, "% Observations Missing: ""{:.2%}".format(loans[col_name].isnull().mean()), color=color, horizontalalignment='left')
    plt.figtext(0.85, 0.90, 'Feature Description:' + feat_desc.loc[0, col_name], wrap=True, backgroundcolor='w',color=color, multialignment='left', horizontalalignment='center')
    plt.savefig(folder_path + "\\" + col_name, dpi=200)
    plt.tight_layout()
#%%Plot all features if show_charts = True
if show_charts is True:
    for col in loans.loc[:, loans.columns !='new_loan_status']:
        try:
            is_continuous = (loans[col].dtype != object)
            plot_var(col, col, continuous=is_continuous)
            pass
        except:
            continue

#%%Feature transformation and creation of custom features
#Convert term feature to integer
loans['term'] = loans['term'].apply(lambda s: np.int8(s.split()[0]))

#Cleans and converts employment length feature
loans['emp_length'].replace(to_replace='n/a', value=np.nan, inplace=True)
loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
loans['emp_length'].replace('< 1 year', '0 years', inplace=True)
def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)

#Converts home ownership feature
loans['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)

#Cleans and converts annual revolving balance feature
#loans['log_revol_bal'] = loans['revol_bal'].apply(lambda x: np.log10(x+1))
#loans.drop('revol_bal', axis=1, inplace=True)

#Cleans and converts annual income feature
#loans['log_annual_inc'] = loans['annual_inc'].apply(lambda x: np.log10(x+1))


#Is loan a specific amount? 1 = Yes, 0 = No
loan_amnt = loans['loan_amnt'].astype('int64')
loan_amnt = loan_amnt.astype(str)
def count_digits(string):
    return sum(item.isdigit() for item in string)
loan_amnt_total = loan_amnt.apply(count_digits)
def trailing_zeros(longint):
    manipulandum = str(longint)
    return len(manipulandum)-len(manipulandum.rstrip('0'))
loan_amnt_zero = loan_amnt.apply(trailing_zeros)
zero_percentage = loan_amnt_zero / loan_amnt_total
loans['specific_loan_amnt'] = (np.where((zero_percentage <= 0.5),1,0)).astype(np.uint8)

#Creates Monthly Loan Payment as % of Monthly Income
loans['monthly_loan_dti'] = (loans['installment']/(loans['annual_inc']/12))

#Convert DTI to decimal
loans['dti'] = loans['dti'] / 100

#Creates increase to DTI feature
loans['dti_increase'] = (loans['loan_amnt']/loans['annual_inc'])
#loans.drop('annual_inc', axis=1, inplace=True)

#Creates credit history feature
loans['credit_history_days'] = (loans['issue_d']-loans['earliest_cr_line']).dt.days
loans.drop('issue_d', axis=1, inplace=True)
loans.drop('earliest_cr_line', axis=1, inplace=True)

#Convert Credit Rating to linear integer
loans['sub_grade'].replace('A1', 1, inplace=True)
loans['sub_grade'].replace('A2', 2, inplace=True)
loans['sub_grade'].replace('A3', 3, inplace=True)
loans['sub_grade'].replace('A4', 4, inplace=True)
loans['sub_grade'].replace('A5', 5, inplace=True)
loans['sub_grade'].replace('B1', 6, inplace=True)
loans['sub_grade'].replace('B2', 7, inplace=True)
loans['sub_grade'].replace('B3', 8, inplace=True)
loans['sub_grade'].replace('B4', 9, inplace=True)
loans['sub_grade'].replace('B5', 10, inplace=True)
loans['sub_grade'].replace('C1', 11, inplace=True)
loans['sub_grade'].replace('C2', 12, inplace=True)
loans['sub_grade'].replace('C3', 13, inplace=True)
loans['sub_grade'].replace('C4', 14, inplace=True)
loans['sub_grade'].replace('C5', 15, inplace=True)
loans['sub_grade'].replace('D1', 16, inplace=True)
loans['sub_grade'].replace('D2', 17, inplace=True)
loans['sub_grade'].replace('D3', 18, inplace=True)
loans['sub_grade'].replace('D4', 19, inplace=True)
loans['sub_grade'].replace('D5', 20, inplace=True)
loans['sub_grade'].replace('E1', 21, inplace=True)
loans['sub_grade'].replace('E2', 22, inplace=True)
loans['sub_grade'].replace('E3', 23, inplace=True)
loans['sub_grade'].replace('E4', 24, inplace=True)
loans['sub_grade'].replace('E5', 25, inplace=True)
loans['sub_grade'].replace('F1', 26, inplace=True)
loans['sub_grade'].replace('F2', 27, inplace=True)
loans['sub_grade'].replace('F3', 28, inplace=True)
loans['sub_grade'].replace('F4', 29, inplace=True)
loans['sub_grade'].replace('F5', 30, inplace=True)
loans['sub_grade'].replace('G1', 31, inplace=True)
loans['sub_grade'].replace('G2', 32, inplace=True)
loans['sub_grade'].replace('G3', 33, inplace=True)
loans['sub_grade'].replace('G4', 34, inplace=True)
loans['sub_grade'].replace('G5', 35, inplace=True)

#Trim the last 3 digits off of Zip Code and convert to integer
loans['zip_code'] = loans['zip_code'].fillna('000xx')
loans['zip_code'] = loans['zip_code'].str[:-2].astype(np.int64)

#Creates fico_score feature by taking the average of low and high fico scores
loans['fico_score'] = 0.5*loans['fico_range_low'] + 0.5*loans['fico_range_high']
loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)

#Convert loan status to 0 or 1
loans['charged_off'] = (loans['new_loan_status'] == 'Charged Off').apply(np.uint8)
loans.drop('new_loan_status', axis=1, inplace=True)

#One hot encode categorical features
onehot_columns = ['home_ownership',
          'verification_status',
          'purpose',
          'addr_state',
          'application_type',
          'initial_list_status']

#Excludes one hot encoded colums from being scaled
columns_temp = loans.columns
scaler_columns = [x for x in columns_temp if x not in onehot_columns]
scaler_columns.remove('PRE_PAY')
scaler_columns.remove('charged_off')
scaler_columns.remove('specific_loan_amnt')
scaler_columns.remove('zip_code')
loans = pd.get_dummies(loans, columns=onehot_columns, drop_first=True)

#%%Split data into Train / Validation / Test
#%%
fully_paid = "Fully Paid: ""{:.0%}".format((loans['charged_off']==0).sum()/len(loans)) #Percentage of loans fully paid
charge_off = "Charged Off: ""{:.0%}".format((loans['charged_off']==1).sum()/len(loans))#Percentage of loans charged off
fully_paid_val = (loans['charged_off']==0).sum()/len(loans) #Used for horizontal line on precision-recall chart
y = loans['charged_off']
X = loans.drop(['charged_off', 'PRE_PAY'], axis=1, inplace=False)
x = X.replace(np.inf, np.nan, inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.10, random_state=1989) #creates train and val splits
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.111111, random_state=1989) #creates val and test splits
print('-----RAW DATASET STATS-----')
print('Train:',"{:,}".format(len(X_train)),"| " "{:.2%}".format(len(X_train)/len(loans)))
print('Val:',"{:,}".format(len(X_val)), "| " "{:.2%}".format(len(X_val) / len(loans)))
print('Test:',"{:,}".format(len(X_test)), "| " "{:.2%}".format(len(X_test) / len(loans)))
print('Total:',"{:,}".format(len(loans)),"| " "{:.2%}".format(len(loans)/len(loans)))
print('-----RAW CLASSES-----')
print(fully_paid)
print(charge_off)

onehot_columns = [x for x in X_train.columns if x not in scaler_columns]
onehot_columns = dict(zip(onehot_columns,range(len(onehot_columns))))
#%%Impute-Scale-SMOTE
#Raw Data = RAW
loans.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\loans_RAW_all.CSV', index=True)
X_train.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TRAIN\loans_RAW_X_train_all.CSV', index=False)
y_train.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TRAIN\loans_RAW_y_train_all.CSV', index=False)
X_val.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\VAL\loans_RAW_X_val_all.CSV', index=True)
y_val.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\VAL\loans_RAW_y_val_all.CSV', index=True)
X_test.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TEST\loans_RAW_X_test_all.CSV', index=True)
y_test.to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TEST\loans_RAW_y_test_all.CSV', index=True)

#Count Classes
fully_paid = "Fully Paid: ""{:.0%}".format((y_train==0).sum()/len(y_train)) #Percentage of loans fully paid
charge_off = "Charged Off: ""{:.0%}".format((y_train==1).sum()/len(y_train))#Percentage of loans charged off

#Print Raw Dataset Stats
print('-----RAW DATASET STATS-----')
print('Train:',"{:,}".format(len(X_train)),"| " "{:.2%}".format(len(X_train)/len(loans)))
print('Val:',"{:,}".format(len(X_val)), "| " "{:.2%}".format(len(X_val) / len(loans)))
print('Test:',"{:,}".format(len(X_test)), "| " "{:.2%}".format(len(X_test) / len(loans)))
print('Total:',"{:,}".format(len(loans)),"| " "{:.2%}".format(len(loans)/len(loans)))
print('-----RAW CLASSES-----')
print(fully_paid)
print(charge_off)

#Save Raw Dataset stats to text file
print('-----RAW DATASET STATS-----', file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Train:',"{:,}".format(len(X_train)),"| " "{:.2%}".format(len(X_train)/len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Val:',"{:,}".format(len(X_val)), "| " "{:.2%}".format(len(X_val) / len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Test:',"{:,}".format(len(X_test)), "| " "{:.2%}".format(len(X_test) / len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Total:',"{:,}".format(len(loans)),"| " "{:.2%}".format(len(loans)/len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('-----RAW CLASSES-----', file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(fully_paid, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(charge_off, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))

#Impute & Standardize Data = IS
columns_train = list(X_train.columns.values) #Columns names to be passed to numpy arrays before saving CSVs
columns_val_test = list(X_val.columns.values) #Columns names to be passed to numpy arrays before saving CSVs
X_train_IS = X_train
X_val_IS = X_val
X_test_IS = X_test

#Impute missing values as mean
imputer = SimpleImputer(missing_values=np.nan, copy=False, strategy='mean')
X_train_IS = pd.DataFrame(imputer.fit_transform(X_train_IS), columns=X_train.columns, index=X_train.index)
X_val_IS = pd.DataFrame(imputer.transform(X_val_IS), columns=X_val.columns, index=X_val.index)
X_test_IS = pd.DataFrame(imputer.transform(X_test_IS), columns=X_test.columns, index=X_test.index)
#Convert data to Z-scores

scaler = StandardScaler()
X_train_IS[scaler_columns] = scaler.fit_transform(X_train_IS[scaler_columns]).astype('float32')
X_val_IS[scaler_columns] = scaler.transform(X_val_IS[scaler_columns]).astype('float32')
X_test_IS[scaler_columns] = scaler.transform(X_test_IS[scaler_columns]).astype('float32')

#Count Classes
fully_paid = "Fully Paid: ""{:.0%}".format((y_train==0).sum()/len(y_train)) #Percentage of loans fully paid
charge_off = "Charged Off: ""{:.0%}".format((y_train==1).sum()/len(y_train))#Percentage of loans charged off

#Save IS Dataset Stats to text file
print('-----IS DATASET STATS-----')
print('Train:',"{:,}".format(len(X_train_IS)),"| " "{:.2%}".format(len(X_train_IS)/len(loans)))
print('Val:',"{:,}".format(len(X_val_IS)), "| " "{:.2%}".format(len(X_val_IS) / len(loans)))
print('Test:',"{:,}".format(len(X_test_IS)), "| " "{:.2%}".format(len(X_test_IS) / len(loans)))
print('Total:',"{:,}".format(len(loans)),"| " "{:.2%}".format(len(loans)/len(loans)))
print('-----IS CLASSES-----')
print(fully_paid)
print(charge_off)

#Save IS Dataset stats to text file
print('-----IS DATASET STATS-----', file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Train:',"{:,}".format(len(X_train_IS)),"| " "{:.2%}".format(len(X_train_IS)/len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Val:',"{:,}".format(len(X_val_IS)), "| " "{:.2%}".format(len(X_val_IS) / len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Test:',"{:,}".format(len(X_test)), "| " "{:.2%}".format(len(X_test_IS) / len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('Total:',"{:,}".format(len(loans)),"| " "{:.2%}".format(len(loans)/len(loans)), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print('-----IS CLASSES-----', file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(fully_paid, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(charge_off, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))

#Save all Imputed & Standardized CSV files
pd.DataFrame(X_train_IS, columns=columns_train).to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TRAIN\loans_IS_X_train_all.CSV', index=False)
pd.DataFrame(X_val_IS).to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\VAL\loans_IS_X_val_all.CSV', index=True)
pd.DataFrame(X_test_IS).to_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TEST\loans_IS_X_test_all.CSV', index=True)

#%%Print final data-set stats
print("----------FINAL LOANS SHAPE----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(loans.shape, file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print("----------LOANS FEATURE DESCRIPTION----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(loans.describe(), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))

#%%IS Linear Dependence of target variable
X = X_train_IS
y = y_train
linear_dep = pd.DataFrame()
for col in X.columns:
    linear_dep.loc[col, 'pearson_corr'] = X[col].corr(y)
linear_dep['abs_pearson_corr'] = abs(linear_dep['pearson_corr'])
for col in X.columns:
    mask = X[col].notnull()
    (linear_dep.loc[col, 'F'], linear_dep.loc[col, 'p_value']) = f_classif(pd.DataFrame(X.loc[mask, col]), y.loc[mask])
linear_dep.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_dep.drop('abs_pearson_corr', axis=1, inplace=True)
linear_dep.reset_index(inplace=True)
linear_dep.rename(columns={'index':'variable'}, inplace=True)
print("----------IS LINEAR DEPENDENCE----------", file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print(linear_dep,file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))

#%%Feature Correlation Matrix
if show_charts is True:
    corr_matrix = loans.corr()
    fig, ax = plt.subplots(figsize=(30, 30), dpi=500)
    sns.heatmap(corr_matrix, square=True, )
    plt.savefig(folder_path + "\\" + 'Correlation Matrix', dpi=500)

#%%Print time to execute
print("--- Total Time to Execute: %s Seconds ---" % (time.time() - start_time), file=open(folder_path+"\\"+"PREPROC_OUTPUT.txt", "a"))
print("--- Total Time to Execute: %s Seconds ---" % (time.time() - start_time))

#%%Send email confirming Script ran correctly
port = 465  # For SSL
smtp_server = "XXXX"
sender_email = "XXXX"  # Enter your address
receiver_email = "XXXX"  # Enter receiver address
password =open(r"XXXX").read()
message = """\
Subject: CL_LOAN_PREPROC Code Executed

This message is sent from MEGA DESK."""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
