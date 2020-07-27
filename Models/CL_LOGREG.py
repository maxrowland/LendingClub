#Author: Max H. Rowland
#Email: maxh.rowland@gmail.com
#Script uses a logistic regression with stochastic gradient descent classifer to predict loan defaults within the lendingClub dataset
#%%General Packages
import os, errno, time, smtplib, ssl, pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from pandas.tseries.offsets import MonthEnd
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
import mysql.connector
from mysql.connector import Error
#Model Specific Packages
from sklearn.linear_model import SGDClassifier
from sklearn import decomposition
#%%Global model output settings
#######################################################################################################################
#######################################################################################################################
model_name = 'Logistic Regression' #Name used in chart titles
save_path = r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\CL_LOGREG' #Directory for saving model output
scorer = 'bal_accuracy_score' #Scoring metric for Grid Search
start_time = time.time() #start time for execution timer
#######################################################################################################################
#######################################################################################################################
#%%Load Train/Val/Test Data Files
#######################################################################################################################
#######################################################################################################################
#Training CSV files
X_train = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TRAIN\loans_IS_X_train_all.CSV')
y_train = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TRAIN\loans_RAW_y_train_all.CSV')

#Validation CSV files
X_val = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\VAL\loans_IS_X_val_all.CSV').set_index('id')
y_val = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\VAL\loans_RAW_y_val_all.CSV').set_index('id')

#Test CSV files
X_test = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TEST\loans_IS_X_test_all.CSV').set_index('id')
y_test = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\TEST\loans_RAW_y_test_all.CSV').set_index('id')
#######################################################################################################################
#######################################################################################################################
#%% Model output save folder creation
timestamp = datetime.today().strftime('%m-%d-%Y_%I-%M %p')
timestamp_label = 'Analysis Date: ' + datetime.today().strftime('%m/%d/%Y %I:%M %p')
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
filecreation(save_path, model_name)
folder_path = save_path + "\\" + timestamp

#%% Logistic Regression with SGD Training
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    #('pca', decomposition.PCA(n_components=5)),
    ('model', SGDClassifier(loss='log', #loss function - default = 'hinge'
                            penalty='l1', #Regularization term - default = 'l2'
                            alpha=.0014, #Constant that multiplies the regularization term - default = 0.0001
                            l1_ratio=0.2, #The Elastic Net mixing parameter - default = 0.15
                            fit_intercept=True, #Whether the intercept should be estimated or not - default = True
                            max_iter= 100, #The maximum number of passes over the training data (aka epochs) - default = 1000
                            tol=0.001, #The stopping criterion - default = 1e-3
                            shuffle=True, #Whether or not the training data should be shuffled after each epoch - default = True
                            verbose=0, #How often progress messages are printed - default = 0
                            epsilon=0.1, #Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’ - default = 0.1
                            n_jobs=8, #Number of CPUs to use - default = None
                            random_state=1989, #The seed of random number generator - default = None
                            learning_rate='optimal', #The learning rate schedule - default = 'optimal'
                            eta0=0.0, #The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules - default = 0.0
                            power_t=0.5, #The exponent for inverse scaling learning rate - default = 0.5
                            early_stopping=False, #Whether to use early stopping to terminate training when validation score is not improving - default = False
                            validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping - default = 0.1
                            n_iter_no_change=5, #Number of iterations with no improvement to wait before early stopping - default = 5
                            class_weight='balanced', #Preset for the class_weight fit parameter - default = None
                            warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. - default = False
                            average=False)#When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute. - default = False
     )
])
param_grid = {
    #'model__loss': ['log'],
    #'model__penalty':['l1','l2'],
    #'model__alpha':[.001, .0011, .0012, .0013, .0014],
    #'model__l1_ratio':[.1, 0.15, .2],
    #'model__fit_intercept':[True],
    #'model__max_iter':[25,50,100,150],
    #'model__tol':[0.01],
    #'model__shuffle':[True],
    #'model__epsilon':[0.1],
    #'model__learning_rate':['optimal'],
    #'model__eta0':[0.01],
    #'model__power_t':[0.5],
    #'model__early_stopping':[False],
    #'model__n_inter_no_change':[5],
    #'model__warm_start': [False],
    #'model__class_weight':[None, 'balanced'],
    #'pca__n_components':[None,2,6,12,20]
}
scorers = {
    'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
    'precision_score': metrics.make_scorer(metrics.precision_score),
    'recall_score': metrics.make_scorer(metrics.recall_score),
    'bal_accuracy_score': metrics.make_scorer(metrics.balanced_accuracy_score)
}
model = GridSearchCV(estimator=pipeline, #Model
                     param_grid=param_grid, #Search grip parameters
                     scoring=scorers, #evaluate the predictions on the test set - default = None
                     n_jobs=8, #Number of CPUs to use - default = None
                     refit=scorer, #Refit an estimator using the best found parameters on the whole dataset. - default = True
                     cv=5, #Determines the cross-validation splitting strategy
                     verbose=2, #How often progress messages are printed - default = 0
                     pre_dispatch='2*n_jobs', #Controls the number of jobs that get dispatched during parallel execution - default = '2*n_jobs'
                     return_train_score=False #If False, the cv_results_ attribute will not include training scores - default = False
                     )
model.fit(X_train, y_train) #Dataset to train the model on
grid_results = pd.DataFrame(model.cv_results_)
grid_results = grid_results.sort_values(by='mean_test_roc_auc', ascending=False)
pickle.dump(model, open(folder_path + "\\" + "MODEL.txt",'wb'))
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#%%Model Output Plots and Stats
#Define passed variables
def model_results(model, model_name, X, y, val_or_test, threshold, window):
    #%%Model result global variables
    ###################################################################################################################
    ###################################################################################################################
    author = 'Max H. Rowland'
    color1 = "#2B115A"  #validation color
    color2 = '#008AC9'  #test color
    color_random = '#f11a21'  #random classifier color
    mblue = "#008AC9"  #chart colors
    mpurple = "#2B115A"  #chart colors
    mred = "#f11a21"  #chart colors
    pub_bm1_color = 'blue'
    pub_bm2_color = 'orange'
    pub_bm3_color = 'purple'
    thresh_label = 'Decision Threshold: {}'.format(threshold)
    classes = ['Fully Paid', 'Charged Off']
    author = 'Max H. Rowland'
    portfolio = 'port_val'
    benchmark = 'bm_val'
    cash = 'BBgBarc US Treasury 1-3 M'
    pub_bm1 = 'S&P 500'
    pub_bm2 = 'BBgBarc US Agg Bond'
    pub_bm3 = 'BBgBarc HY 1-3 Yr Ba'
    #Axis Formatting
    percent_0 = mtick.PercentFormatter(xmax=1, decimals=0, symbol='%')
    percent_1 = mtick.PercentFormatter(xmax=1, decimals=1, symbol='%')
    percent_2 = mtick.PercentFormatter(xmax=1, decimals=2, symbol='%')
    dollars = mtick.StrMethodFormatter('${x:,.0f}')
    commas = mtick.StrMethodFormatter('{x:,.0f}')

    #Creates Proper Test or Validation Titles
    if val_or_test is 'val':
        val_or_test_label = 'VALIDATION'
    else: val_or_test_label = 'TEST'

    #Loans and Public Market Returns
    #loans Raw CSV files
    loans = pd.read_csv(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\loans_RAW_all.CSV').set_index('id')
    #Load Index Returns
    mkt_ret = pd.read_excel(r'E:\DATABASES\LENDING_CLUB\MODEL DATA\ECON_DATA.XLSX', sheet_name='RETURNS', index_col=0, skiprows=2)
    ###################################################################################################################
    ###################################################################################################################

    #Query model on Validation and Test data then convert to Classes based on threshold
    y_val_score = model.predict_proba(X)[:, 1]  #Probability estimates 0 to 1

    def adjusted_classes(y_score, threshold):
        return [1 if y >= threshold else 0 for y in y_score]
    y_val_pred = adjusted_classes(y_val_score, threshold) #Converts probability estimates to 0 or 1 based on Threshold

    if val_or_test is 'val':
        index = y_val.index
    else: index = y_test.index

    y_val_score = pd.DataFrame(y_val_score, index=index)
    y_val_pred = pd.DataFrame(y_val_pred, index=index)
    y_val_pred.rename(columns={0:'charged_off'}, inplace=True)

    #Histograph of Charge-off Scores
    plt.hist(y_val_score[0], bins=50, histtype='bar', ec='black', facecolor=mblue)
    plt.axvline(threshold, ymin=0, ymax=1, linestyle='--', color=color_random,
                label='Current Threshold: %0.2f' % threshold)
    plt.title('Scores')
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig(folder_path + "\\" + 'Scores', dpi=200)
    plt.show()

    #%%MySQL query for loan returns

    #Creates a list of loans in the Validation dataset
    val_loans_list = y_val_pred.index.tolist()
    val_loans_list = [str(item) for item in val_loans_list]


    try:
        cnx = mysql.connector.connect(host='XXXX',
                                       port=XXXX,
                                       database='XXXX',
                                       user='XXXX',
                                       password='XXXX')

#%%Returns
        #Validation Returns Query
        format_strings = ','.join(['%s'] * len(val_loans_list))
        cur = cnx.cursor()
        cur.execute('SELECT * FROM lendingclub.loans_returns WHERE LOAN_ID IN (%s)' % format_strings,
                     tuple(val_loans_list))
        val_loans_ret = cur.fetchall()

        #Create DataFrame & apply column names
        val_loans_ret = pd.DataFrame(val_loans_ret, columns=cur.column_names)

#%%Characteristics
    #Validation Characteristics Query
        format_strings = ','.join(['%s'] * len(val_loans_list))
        cur = cnx.cursor()
        cur.execute(
            'SELECT id, grade, dti, term, int_rate, loan_amnt, fico_range_low, fico_range_high, purpose FROM lendingclub.loans_all WHERE id IN (%s)' % format_strings,
            tuple(val_loans_list))
        val_loans = cur.fetchall()

        #Create DataFrame & apply column names
        val_loans = pd.DataFrame(val_loans, columns=cur.column_names)

#%%Pre-Pay
    #Validation Pre-Pay Query
        format_strings = ','.join(['%s'] * len(val_loans_list))
        cur = cnx.cursor()
        cur.execute('SELECT * FROM lendingclub.loans_pre_pay WHERE LOAN_ID IN (%s)' % format_strings,
                    tuple(val_loans_list))
        val_loans_pre_pay = cur.fetchall()

        #Create DataFrame & apply column names
        val_loans_pre_pay = pd.DataFrame(val_loans_pre_pay, columns=cur.column_names)
#%%Charge-Off
    #Validation Charge-Off Query
        format_strings = ','.join(['%s'] * len(val_loans_list))
        cur = cnx.cursor()
        cur.execute('SELECT * FROM lendingclub.loans_charge_off WHERE LOAN_ID IN (%s)' % format_strings,
                    tuple(val_loans_list))
        val_loans_charge_off = cur.fetchall()

        #Create DataFrame & apply column names
        val_loans_charge_off = pd.DataFrame(val_loans_charge_off, columns=cur.column_names)


    #Closes MySQL Connection or Displays Connection Error
    except Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if cnx.is_connected():
            cnx.close()
            cur.close()
            print("MySQL Connection is Closed")
            print("--- MySQL Download Time: %s seconds ---" % (time.time() - start_time))

    #%%Data transformations on MySQL Queries
    #Transform MONTH field and load
    val_loans_ret['MONTH'] = val_loans_ret['MONTH'].astype('datetime64[M]') - pd.tseries.offsets.MonthEnd(0)
    val_loans_pre_pay['MONTH'] = val_loans_pre_pay['MONTH'].astype('datetime64[M]') - pd.tseries.offsets.MonthEnd(0)
    val_loans_charge_off['MONTH'] = val_loans_charge_off['MONTH'].astype('datetime64[M]') - pd.tseries.offsets.MonthEnd(0)

    #pivot validation loan returns dataframe
    val_loans_ret = val_loans_ret.sort_values(by='MONTH', ascending=True)
    val_loans_ret = val_loans_ret.pivot_table(values='NET_RETURN', index='MONTH', columns='LOAN_ID')

    #pivot validation pre-pay dataframe
    val_loans_pre_pay = val_loans_pre_pay.sort_values(by='MONTH', ascending=True)
    val_loans_pre_pay = val_loans_pre_pay.pivot_table(values='PRE_PAY', index='MONTH', columns='LOAN_ID')

    #pivot validation pre-pay dataframe
    val_loans_charge_off = val_loans_charge_off.sort_values(by='MONTH', ascending=True)
    val_loans_charge_off = val_loans_charge_off.pivot_table(values='CHARGE_OFF', index='MONTH', columns='LOAN_ID')


    #Calculate Average Fico Score and drop high and low columns
    val_loans['fico_score'] = 0.5 * val_loans['fico_range_low'] + 0.5 * val_loans['fico_range_high']
    val_loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)

    #Convert DTI to percentage
    val_loans['dti'] = val_loans['dti'] / 100

    #Convert term to integers
    val_loans['term'] = val_loans['term'].apply(lambda s: np.int8(s.split()[0]))

    #Benchmark weights
    bm_val_n_loans = val_loans_ret.apply(lambda x: x.count(), axis=1)
    bm_val_weights = 1 / bm_val_n_loans
    bm_val = val_loans_ret.mul(bm_val_weights, axis='index').sum(axis=1).rename('bm_val')

    #Create return dataframe
    bm_val.index = pd.to_datetime(bm_val.index)
    bm_val_start = bm_val.index.min()  #Start of benchmark data
    bm_val_end = bm_val.index.max()  #End of benchmark data
    val_date_range = 'Returns Date Range: ' + bm_val_start.strftime('%m/%d/%Y') + ' to ' + bm_val_end.strftime('%m/%d/%Y')#Table Date Span Text
    val_ret_index = pd.date_range(start=bm_val_start, end=bm_val_end, freq='M')  #create datetime index from start to end dates

    #Create return dataframe
    val_ret = pd.DataFrame(index=val_ret_index)
    val_ret = pd.merge(val_ret, mkt_ret, left_index=True, right_index=True)  #merge public market returns to ret dataframe
    val_ret = val_ret.merge(bm_val, how='left', left_index=True, right_index=True).fillna(0)  #merge equal weight benchmark to ret dataframe

    #Portfolio list and returns, pre-pay, and charge-off dataframes
    port_val_loans_list = y_val_pred[y_val_pred['charged_off'] == 0].index.tolist()
    port_val_loans_list = [str(item) for item in port_val_loans_list]
    port_val_loans_ret = val_loans_ret.filter(items=port_val_loans_list, axis=1)
    port_val_loans_pre_pay = val_loans_pre_pay.filter(items=port_val_loans_list, axis=1)
    port_val_loans_charge_off = val_loans_charge_off.filter(items=port_val_loans_list, axis=1)

    #Portfolio weights
    port_val_n_loans = port_val_loans_ret.apply(lambda x: x.count(), axis=1)
    port_val_weights = 1 / port_val_n_loans
    port_val = port_val_loans_ret.mul(port_val_weights, axis='index').sum(axis=1).rename('port_val')

    #Merge equal weight portfolio to ret dataframe
    val_ret = val_ret.merge(port_val, how='left', left_index=True, right_index=True).fillna(0)

    #Pivot validation loans dataframe
    val_loans = val_loans.pivot_table(columns='id', aggfunc='first')
    port_val_loans = val_loans.filter(items=port_val_loans_list, axis=1)

    #Unstack True predictions and add them to val_loans DF
    y_val_unstack = pd.pivot_table(y_val, values='charged_off', columns='id')
    y_val_unstack.columns = y_val_unstack.columns.astype(str)
    val_loans = pd.concat([y_val_unstack, val_loans])
    port_val_loans = pd.concat([y_val_unstack, port_val_loans])

    #%%Data-set information
    #Counts for annotations and Text File output
    total = len(X_train) + len(X_val) + len(X_test)
    fully_paid_train = "Train Fully Paid: ""{:.0%}".format(
        (y_train['charged_off'] == 0).sum() / len(y_train))  # Percentage of loans fully paid
    charge_off_train = "Train Charged Off: ""{:.0%}".format(
        (y_train['charged_off'] == 1).sum() / len(y_train))  # Percentage of loans charged off

    fully_paid_val = "Val Fully Paid: ""{:.0%}".format((y_val['charged_off'] == 0).sum() / len(y_val))
    charge_off_val = "Val Charged Off: ""{:.0%}".format((y_val['charged_off'] == 1).sum() / len(y_val))

    fully_paid_test = "Test Fully Paid: ""{:.0%}".format((y_test['charged_off'] == 0).sum() / len(y_test))
    charge_off_test = "Test Charged Off: ""{:.0%}".format((y_test['charged_off'] == 1).sum() / len(y_test))
    ttv_footnote = "Train: ""{:,}".format(len(X_train)) + \
                   "|" "{:.0%}".format(len(X_train) / total) + \
                   " - ""Val: ""{:,}".format(len(X_val)) + \
                   "|""{:.0%}".format(len(X_val) / total) + \
                   " - ""Test: ""{:,}".format(len(X_test)) + \
                   "|""{:.0%}".format(len(X_test) / total)

    # Counts for Plots
    train_total_plt = len(X_train)

    train_fully_paid_plt = (y_train['charged_off'] == 0).sum() / len(y_train)
    train_charge_off_plt = (y_train['charged_off'] == 1).sum() / len(y_train)

    val_total_plt = len(X_val)
    val_fully_paid_plt = (y_val['charged_off'] == 0).sum() / len(y_val)
    val_charge_off_plt = (y_val['charged_off'] == 1).sum() / len(y_val)

    test_total_plt = len(X_test)
    test_fully_paid_plt = (y_test['charged_off'] == 0).sum() / len(y_test)
    test_charge_off_plt = (y_test['charged_off'] == 1).sum() / len(y_test)

    print('----------DATASET STATS----------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Train:', "{:,}".format(len(X_train)), "| " "{:.2%}".format(len(X_train) / total),
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Val:', "{:,}".format(len(X_val)), "| " "{:.2%}".format(len(X_val) / total),
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Test:', "{:,}".format(len(X_test)), "| " "{:.2%}".format(len(X_test) / total),
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Total:', "{:,}".format(len(loans)), "| " "{:.2%}".format(len(loans) / total),
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('----------CLASSES----------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(fully_paid_train, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(charge_off_train, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(fully_paid_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(charge_off_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(fully_paid_test, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(charge_off_test, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('----------THRESHOLD----------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(threshold, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))

    #%%Calculate Validation Model Stats
    f1_val = metrics.f1_score(y, y_val_pred, average=None)
    f1_micro_val = metrics.f1_score(y, y_val_pred, average='micro')
    f1_macro_val = metrics.f1_score(y, y_val_pred, average='macro')
    f1_weighted_val = metrics.f1_score(y, y_val_pred, average='weighted', labels=[0, 1])
    accuracy_val = metrics.accuracy_score(y, y_val_pred, normalize=True)
    balanced_accuracy_val = metrics.balanced_accuracy_score(y, y_val_pred, adjusted=False)  #adjusted: 1=best 0=worst
    average_precision_val = metrics.average_precision_score(y, y_val_pred)
    precision_val = metrics.precision_score(y, y_val_pred, average=None)
    precision_micro_val = metrics.precision_score(y, y_val_pred, average='micro')
    precision_macro_val = metrics.precision_score(y, y_val_pred, average='macro')
    precision_weighted_val = metrics.precision_score(y, y_val_pred, average='weighted')
    recall_val = metrics.recall_score(y, y_val_pred, average=None)
    recall_micro_val = metrics.recall_score(y, y_val_pred, average='micro')
    recall_macro_val = metrics.recall_score(y, y_val_pred, average='macro')
    recall_weighted_val = metrics.recall_score(y, y_val_pred, average='weighted')
    jaccard_val = metrics.jaccard_score(y, y_val_pred, average=None)
    jaccard_micro_val = metrics.jaccard_score(y, y_val_pred, average='micro')
    jaccard_macro_val = metrics.jaccard_score(y, y_val_pred, average='macro')
    jaccard_weighted_val = metrics.jaccard_score(y, y_val_pred, average='weighted')
    roc_auc_val = metrics.roc_auc_score(y, y_val_score, average=None)
    roc_auc_micro_val = metrics.roc_auc_score(y, y_val_score, average='micro')
    roc_auc_macro_val = metrics.roc_auc_score(y, y_val_score, average='macro')
    roc_auc_weighted_val = metrics.roc_auc_score(y, y_val_score, average='weighted')
    roc_auc_samples_val = metrics.roc_auc_score(y, y_val_score, average='samples')
    brier_score_val = metrics.brier_score_loss(y, y_val_score)
    neg_log_loss_val = metrics.log_loss(y, y_val_pred)
    mmc_val = metrics.matthews_corrcoef(y, y_val_pred)
    cm_val = metrics.confusion_matrix(y, y_val_pred)
    cmn_val = metrics.confusion_matrix(y, y_val_pred, normalize='all')
    cr_val = metrics.classification_report(y, y_val_pred, output_dict=False, target_names=classes)

    #Print Validation Stats to Text File
    print('----------'+val_or_test_label+' SUMMARY RESULTS----------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(cr_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('MICRO = Calculate metrics globally by counting the total true positives, false negatives and false positives. ',
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('MACRO = Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account. ',
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('WEIGHTED = Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).',
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(cm_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------NORMALIZED CONFUSION MATRIX---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print(cmn_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------f1 METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('f1 = Harmonic mean of precision and recall - 0= worst 1= best ', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('f1: ', f1_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('f1 Micro: '"{:0.4}".format(f1_micro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('f1 Macro: '"{:0.4}".format(f1_macro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('f1 Weighted: '"{:0.4}".format(f1_weighted_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------ACCURACY METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ACCURACY = (TP+TN)/(TP+TN+FP+FN)', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Accuracy: '"{:0.4}".format(accuracy_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Balanced Accuracy: '"{:0.4}".format(balanced_accuracy_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------PRECISION METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('PRECISION = TP/(TP+FP)', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Average Precision: '"{:0.4}".format(average_precision_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Precision: ', precision_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Precision Micro: '"{:0.4}".format(precision_micro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Precision Macro: '"{:0.4}".format(precision_macro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Precision Weighted: '"{:0.4}".format(precision_weighted_val),file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------RECALL METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('RECALL = TP/(TP+FN)', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Recall: ', recall_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Recall Micro: '"{:0.4}".format(recall_micro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Recall Macro: '"{:0.4}".format(recall_macro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Recall Weighted: '"{:0.4}".format(recall_weighted_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------JACCARD METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('JACCARD = defined as the size of the intersection divided by the size of the union of two label sets',
          file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Jaccard: ', jaccard_val, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Jaccard Micro: '"{:0.4}".format(jaccard_micro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Jaccard Macro: '"{:0.4}".format(jaccard_macro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Jaccard Weighted: '"{:0.4}".format(jaccard_weighted_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------ROC METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC = Area Under the Receiver Operating Characteristic Curve', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC AUC: '"{:0.4}".format(roc_auc_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC AUC Micro: '"{:0.4}".format(roc_auc_micro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC AUC Macro: '"{:0.4}".format(roc_auc_macro_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC AUC Weighted: '"{:0.4}".format(roc_auc_weighted_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('ROC AUC Samples: '"{:0.4}".format(roc_auc_samples_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------OTHER METRICS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Brier Score (lower=better): '"{:0.4}".format(brier_score_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Cross-entropy Loss (lower=better): '"{:0.4}".format(neg_log_loss_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('MMC: '"{:0.4}".format(mmc_val), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------GRIDSEARCH SCORE---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('GridSearchCV Scorer: ', scorer, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('GridSearchCV Score: '"{:0.4}".format(model.best_score_), file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------GRIDSEARCH PARAMETERS---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('GridSearchCV Hyperparameters: ', model.best_params_, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------BEST GRIDSEARCH MODEL---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('GridSearchCV Best Estimator', model.best_estimator_, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('---------CURRENT MODEL---------', file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print('Pipeline: ', pipeline, file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    #%%Return Calculations
    #Performance
    ret_cum = (1 + val_ret).cumprod() - 1

    #Drawdown
    mdd_cum = (1 + val_ret).cumprod()
    running_max = np.maximum.accumulate(mdd_cum)
    running_max[running_max < 1] = 1
    drawdown = (mdd_cum) / running_max - 1

    # Excess Performance
    excess_ret_cum = (1 + val_ret.sub(val_ret[benchmark], axis=0)).cumprod() - 1

    #Rolling
    #Return
    if window > 12:
        ret_rolling = (1 + val_ret).rolling(window=window).apply(np.prod) ** (12 / window) - 1
    else:
        ret_rolling = (1 + val_ret).rolling(window=window).apply(np.prod) - 1

    #Excess Return
    excess_ret_rolling = ret_rolling.sub(ret_rolling[benchmark], axis=0)

    #Standard Deviation
    std_rolling = val_ret.rolling(window=window).std() * np.sqrt(12)

    #Tracking Error
    excess_ret = val_ret.sub(val_ret[benchmark], axis=0)
    te_rolling = excess_ret.rolling(window=window).std() * np.sqrt(12)

    #Sharpe Ratio
    sr_rolling = (ret_rolling.sub(ret_rolling[cash], axis=0)) / std_rolling

    #Information Ratio
    ir_rolling = excess_ret_rolling / te_rolling

    #Beta
    cov = val_ret.rolling(window=window).cov(val_ret[benchmark])
    var = val_ret[benchmark].rolling(window=window).var()
    beta_rolling = cov.div(var, axis=0)

    #Correlation
    corr_rolling = val_ret.rolling(window=window).corr(val_ret[benchmark])

    #Alpha
    alpha_rolling = ret_rolling - beta_rolling.mul((ret_rolling[benchmark].sub(ret_rolling[cash], axis=0)), axis=0).add(ret_rolling[cash], axis=0)

    #Point Estimate Stats
    stats = pd.DataFrame()  #create empty stats dataframe
    #Create Trailing dataframes
    #1 Year
    ret_1yr = val_ret.tail(12)
    #3 Year
    ret_3yr = val_ret.tail(36)
    #5 Year
    ret_5yr = val_ret.tail(60)
    #10 Year
    ret_10yr = val_ret.tail(120)

    #Return
    ret_1yr_trail = (1 + ret_1yr).apply(np.prod) - 1
    ret_1yr_trail = ret_1yr_trail.rename('1_yr_return')
    stats = stats.append(ret_1yr_trail)

    ret_3yr_trail = (1 + ret_3yr).apply(np.prod) ** (1 / 3) - 1
    ret_3yr_trail = ret_3yr_trail.rename('3_yr_return')
    stats = stats.append(ret_3yr_trail)

    ret_5yr_trail = (1 + ret_5yr).apply(np.prod) ** (1 / 5) - 1
    ret_5yr_trail = ret_5yr_trail.rename('5_yr_return')
    stats = stats.append(ret_5yr_trail)

    ret_10yr_trail = (1 + ret_10yr).apply(np.prod) ** (1 / 10) - 1
    ret_10yr_trail = ret_10yr_trail.rename('10_Yr_return')
    stats = stats.append(ret_10yr_trail)

    ret_incep = (1 + val_ret).apply(np.prod) ** (12 / len(val_ret)) - 1
    ret_incep = ret_incep.rename('incep_return')
    stats = stats.append(ret_incep)

    #Excess Return
    excess_ret_1yr_trail = ret_1yr_trail.sub(ret_1yr_trail[benchmark], axis=0)
    excess_ret_1yr_trail = excess_ret_1yr_trail.rename('1_yr_excess')
    stats = stats.append(excess_ret_1yr_trail)

    excess_ret_3yr_trail = ret_3yr_trail.sub(ret_3yr_trail[benchmark], axis=0)
    excess_ret_3yr_trail = excess_ret_3yr_trail.rename('3_yr_excess')
    stats = stats.append(excess_ret_3yr_trail)

    excess_ret_5yr_trail = ret_5yr_trail.sub(ret_5yr_trail[benchmark], axis=0)
    excess_ret_5yr_trail = excess_ret_5yr_trail.rename('5_yr_excess')
    stats = stats.append(excess_ret_5yr_trail)

    excess_ret_10yr_trail = ret_10yr_trail.sub(ret_10yr_trail[benchmark], axis=0)
    excess_ret_10yr_trail = excess_ret_10yr_trail.rename('10_yr_excess')
    stats = stats.append(excess_ret_10yr_trail)

    excess_ret_incep = ret_incep.sub(ret_incep[benchmark], axis=0)
    excess_ret_incep = excess_ret_incep.rename('incep_excess')
    stats = stats.append(excess_ret_incep)

    #Standard Deviation
    std_3yr_trail = ret_3yr.std() * np.sqrt(12)
    std_3yr_trail = std_3yr_trail.rename('3_yr_std')
    stats = stats.append(std_3yr_trail)

    std_5yr_trail = ret_5yr.std() * np.sqrt(12)
    std_5yr_trail = std_5yr_trail.rename('5_yr_std')
    stats = stats.append(std_5yr_trail)

    std_10yr_trail = ret_10yr.std() * np.sqrt(12)
    std_10yr_trail = std_10yr_trail.rename('10_yr_std')
    stats = stats.append(std_10yr_trail)

    std_incep = val_ret.std() * np.sqrt(12)
    std_incep = std_incep.rename('incep_std')
    stats = stats.append(std_incep)

    #Tracking Error
    #Create Trailing Excess Dataframes
    excess_ret_3yr = ret_3yr.sub(ret_3yr[benchmark], axis=0)
    excess_ret_5yr = ret_5yr.sub(ret_5yr[benchmark], axis=0)
    excess_ret_10yr = ret_10yr.sub(ret_10yr[benchmark], axis=0)

    te_3yr_trail = excess_ret_3yr.std() * np.sqrt(12)
    te_3yr_trail = te_3yr_trail.rename('3_yr_te')
    stats = stats.append(te_3yr_trail)

    te_5yr_trail = excess_ret_5yr.std() * np.sqrt(12)
    te_5yr_trail = te_5yr_trail.rename('5_yr_te')
    stats = stats.append(te_5yr_trail)

    te_10yr_trail = excess_ret_10yr.std() * np.sqrt(12)
    te_10yr_trail = te_10yr_trail.rename('10_yr_te')
    stats = stats.append(te_10yr_trail)

    te_incep = excess_ret.std() * np.sqrt(12)
    te_incep = te_incep.rename('incep_te')
    stats = stats.append(te_incep)

    #Sharpe Ratio
    sr_3yr_trail = (ret_3yr_trail.sub(ret_3yr_trail[cash], axis=0)) / std_3yr_trail
    sr_3yr_trail = sr_3yr_trail.rename('3_yr_sharpe')
    stats = stats.append(sr_3yr_trail)

    sr_5yr_trail = (ret_5yr_trail.sub(ret_5yr_trail[cash], axis=0)) / std_5yr_trail
    sr_5yr_trail = sr_5yr_trail.rename('5_yr_sharpe')
    stats = stats.append(sr_5yr_trail)

    sr_10yr_trail = (ret_10yr_trail.sub(ret_10yr_trail[cash], axis=0)) / std_10yr_trail
    sr_10yr_trail = sr_10yr_trail.rename('10_yr_sharpe')
    stats = stats.append(sr_10yr_trail)

    sr_incep = (ret_incep.sub(ret_incep[cash], axis=0)) / std_incep
    sr_incep = sr_incep.rename('incep_sharpe')
    stats = stats.append(sr_incep)

    #Information Ratio
    ir_3yr_trail = excess_ret_3yr_trail / te_3yr_trail
    ir_3yr_trail = ir_3yr_trail.rename('3_yr_info')
    stats = stats.append(ir_3yr_trail)

    ir_5yr_trail = excess_ret_5yr_trail / te_5yr_trail
    ir_5yr_trail = ir_5yr_trail.rename('5_yr_info')
    stats = stats.append(ir_5yr_trail)

    ir_10yr_trail = excess_ret_10yr_trail / te_10yr_trail
    ir_10yr_trail = ir_10yr_trail.rename('10_yr_info')
    stats = stats.append(ir_10yr_trail)

    ir_incep = excess_ret_incep / te_incep
    ir_incep = ir_incep.rename('incep_info')
    stats = stats.append(ir_incep)

    #Beta
    #3 Year
    cov3 = ret_3yr.cov()
    var3 = ret_3yr[benchmark].var()

    #5 Year
    cov5 = ret_5yr.cov()
    var5 = ret_5yr[benchmark].var()

    #10 Year
    cov10 = ret_10yr.cov()
    var10 = ret_10yr[benchmark].var()

    #Inception
    cov_incep = val_ret.cov()
    var_incep = val_ret[benchmark].var()

    beta_3yr_trail = cov3.div(var3, axis=0)
    stats = stats.append(beta_3yr_trail[benchmark].rename('3_yr_beta'))

    beta_5yr_trail = cov5.div(var5, axis=0)
    stats = stats.append(beta_5yr_trail[benchmark].rename('5_yr_beta'))

    beta_10yr_trail = cov10.div(var10, axis=0)
    stats = stats.append(beta_10yr_trail[benchmark].rename('10_yr_beta'))

    beta_incep = cov_incep.div(var_incep, axis=0)
    stats = stats.append(beta_incep[benchmark].rename('incep_beta'))

    #Correlation
    corr_3yr_trail = ret_3yr.corr()
    stats = stats.append(corr_3yr_trail[benchmark].rename('3_yr_r2'))

    corr_5yr_trail = ret_5yr.corr()
    stats = stats.append(corr_5yr_trail[benchmark].rename('5_yr_r2'))

    corr_10yr_trail = ret_10yr.corr()
    stats = stats.append(corr_10yr_trail[benchmark].rename('10_yr_r2'))

    corr_incep = val_ret.corr()
    stats = stats.append(corr_incep[benchmark].rename('incep_r2'))

    #Alpha
    alpha_3yr_trail = ret_3yr_trail - (
                ret_3yr_trail[cash] + beta_3yr_trail[benchmark] * (ret_3yr_trail[benchmark] - ret_3yr_trail[cash]))
    alpha_3yr_trail = alpha_3yr_trail.rename('3_yr_alpha')
    stats = stats.append(alpha_3yr_trail)

    alpha_5yr_trail = ret_5yr_trail - (
                ret_5yr_trail[cash] + beta_5yr_trail[benchmark] * (ret_5yr_trail[benchmark] - ret_5yr_trail[cash]))
    alpha_5yr_trail = alpha_5yr_trail.rename('5_yr_alpha')
    stats = stats.append(alpha_5yr_trail)

    alpha_10yr_trail = ret_10yr_trail - (
                ret_10yr_trail[cash] + beta_10yr_trail[benchmark] * (ret_10yr_trail[benchmark] - ret_10yr_trail[cash]))
    alpha_10yr_trail = alpha_10yr_trail.rename('10_yr_alpha')
    stats = stats.append(alpha_10yr_trail)

    alpha_incep = ret_incep - (ret_incep[cash] + beta_incep[benchmark] * (ret_incep[benchmark] - ret_incep[cash]))
    alpha_incep = alpha_incep.rename('incep_alpha')
    stats = stats.append(alpha_incep)

    #Max Drawdown
    max_dd = drawdown.agg(np.min)
    max_dd = max_dd.rename('max_drawdown')
    stats = stats.append(max_dd)

    #Min / Max / Average
    min = val_ret.agg(np.min)
    min = min.rename('min_return')
    stats = stats.append(min)

    max = val_ret.agg(np.max)
    max = max.rename('max_return')
    stats = stats.append(max)

    mean = val_ret.agg(np.mean)
    mean = mean.rename('mean_return')
    stats = stats.append(mean)

    #%%Characteristics Calculations
    #interest rate - 'int_rate'
    bm_val_int_rate = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_int_rate = bm_val_int_rate.where(val_loans_ret.isnull(), val_loans.loc['int_rate'], axis=1)
    bm_val_int_rate = bm_val_int_rate.mul(bm_val_weights, axis='index').sum(axis=1)

    port_val_int_rate = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_int_rate = port_val_int_rate.where(port_val_loans_ret.isnull(), port_val_loans.loc['int_rate'], axis=1)
    port_val_int_rate = port_val_int_rate.mul(port_val_weights, axis='index').sum(axis=1)

    #Average FICO - 'fico_score'
    bm_val_fico = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_fico = bm_val_fico.where(val_loans_ret.isnull(), val_loans.loc['fico_score'], axis=1)
    bm_val_fico = bm_val_fico.mul(bm_val_weights, axis='index').sum(axis=1)

    port_val_fico = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_fico = port_val_fico.where(port_val_loans_ret.isnull(), port_val_loans.loc['fico_score'], axis=1)
    port_val_fico = port_val_fico.mul(port_val_weights, axis='index').sum(axis=1)

    #loan amount - 'loan_amnt'
    bm_val_loan_amnt = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_loan_amnt = bm_val_loan_amnt.where(val_loans_ret.isnull(), val_loans.loc['loan_amnt'], axis=1)
    bm_val_loan_amnt = bm_val_loan_amnt.mul(bm_val_weights, axis='index').sum(axis=1)

    port_val_loan_amnt = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_loan_amnt = port_val_loan_amnt.where(port_val_loans_ret.isnull(), port_val_loans.loc['loan_amnt'], axis=1)
    port_val_loan_amnt = port_val_loan_amnt.mul(port_val_weights, axis='index').sum(axis=1)

    #DTI - 'dti'
    bm_val_dti = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_dti = bm_val_dti.where(val_loans_ret.isnull(), val_loans.loc['dti'], axis=1)
    bm_val_dti = bm_val_dti.mul(bm_val_weights, axis='index').sum(axis=1)

    port_val_dti = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_dti = port_val_dti.where(port_val_loans_ret.isnull(), port_val_loans.loc['dti'], axis=1)
    port_val_dti = port_val_dti.mul(port_val_weights, axis='index').sum(axis=1)

    #Average term
    bm_val_term = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_term = bm_val_term.where(val_loans_ret.isnull(), val_loans.loc['term'], axis=1)
    bm_val_term_avg = bm_val_term.mul(bm_val_weights, axis='index').sum(axis=1)

    port_val_term = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_term = port_val_term.where(port_val_loans_ret.isnull(), port_val_loans.loc['term'], axis=1)
    port_val_term_avg = port_val_term.mul(port_val_weights, axis='index').sum(axis=1)

    #Term Allocation
    bm_val_term_36 = bm_val_term.eq(36).sum(axis=1).rename('bm_36')
    bm_val_term_60 = bm_val_term.eq(60).sum(axis=1).rename('bm_60')
    bm_val_term_total = bm_val_term_36 + bm_val_term_60
    bm_val_term_36 = bm_val_term_36 / bm_val_term_total
    bm_val_term_60 = bm_val_term_60 / bm_val_term_total

    port_val_term_36 = port_val_term.eq(36).sum(axis=1).rename('port_36')
    port_val_term_60 = port_val_term.eq(60).sum(axis=1).rename('port_60')
    port_val_term_total = port_val_term_36 + port_val_term_60
    port_val_term_36 = port_val_term_36 / port_val_term_total
    port_val_term_60 = port_val_term_60 / port_val_term_total

    #Benchmark Average Credit
    bm_val_credit = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_credit = bm_val_credit.where(val_loans_ret.isnull(), val_loans.loc['grade'], axis=1)
    bm_val_credit = bm_val_credit.replace('A', 1)
    bm_val_credit = bm_val_credit.replace('B', 2)
    bm_val_credit = bm_val_credit.replace('C', 3)
    bm_val_credit = bm_val_credit.replace('D', 4)
    bm_val_credit = bm_val_credit.replace('E', 5)
    bm_val_credit = bm_val_credit.replace('F', 6)
    bm_val_credit = bm_val_credit.replace('G', 7)
    bm_val_credit_avg = bm_val_credit.mul(bm_val_weights, axis='index').sum(axis=1)

    #Benchmark Credit Allocation
    bm_val_credit_a = bm_val_credit.eq(1).sum(axis=1).rename('bm_a')
    bm_val_credit_b = bm_val_credit.eq(2).sum(axis=1).rename('bm_b')
    bm_val_credit_c = bm_val_credit.eq(3).sum(axis=1).rename('bm_c')
    bm_val_credit_d = bm_val_credit.eq(4).sum(axis=1).rename('bm_d')
    bm_val_credit_e = bm_val_credit.eq(5).sum(axis=1).rename('bm_e')
    bm_val_credit_f = bm_val_credit.eq(6).sum(axis=1).rename('bm_f')
    bm_val_credit_g = bm_val_credit.eq(7).sum(axis=1).rename('bm_g')

    bm_val_credit_total = (bm_val_credit_a +
                           bm_val_credit_b +
                           bm_val_credit_c +
                           bm_val_credit_d +
                           bm_val_credit_e +
                           bm_val_credit_f +
                           bm_val_credit_g)

    bm_val_credit_a = bm_val_credit_a / bm_val_credit_total
    bm_val_credit_b = bm_val_credit_b / bm_val_credit_total
    bm_val_credit_c = bm_val_credit_c / bm_val_credit_total
    bm_val_credit_d = bm_val_credit_d / bm_val_credit_total
    bm_val_credit_e = bm_val_credit_e / bm_val_credit_total
    bm_val_credit_f = bm_val_credit_f / bm_val_credit_total
    bm_val_credit_g = bm_val_credit_g / bm_val_credit_total

    #Portfolio Average Credit
    port_val_credit = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_credit = port_val_credit.where(port_val_loans_ret.isnull(), port_val_loans.loc['grade'], axis=1)
    port_val_credit = port_val_credit.replace('A', 1)
    port_val_credit = port_val_credit.replace('B', 2)
    port_val_credit = port_val_credit.replace('C', 3)
    port_val_credit = port_val_credit.replace('D', 4)
    port_val_credit = port_val_credit.replace('E', 5)
    port_val_credit = port_val_credit.replace('F', 6)
    port_val_credit = port_val_credit.replace('G', 7)
    port_val_credit_avg = port_val_credit.mul(port_val_weights, axis='index').sum(axis=1)

    #Portfolio Credit Allocation
    port_val_credit_a = port_val_credit.eq(1).sum(axis=1).rename('port_a')
    port_val_credit_b = port_val_credit.eq(2).sum(axis=1).rename('port_b')
    port_val_credit_c = port_val_credit.eq(3).sum(axis=1).rename('port_c')
    port_val_credit_d = port_val_credit.eq(4).sum(axis=1).rename('port_d')
    port_val_credit_e = port_val_credit.eq(5).sum(axis=1).rename('port_e')
    port_val_credit_f = port_val_credit.eq(6).sum(axis=1).rename('port_f')
    port_val_credit_g = port_val_credit.eq(7).sum(axis=1).rename('port_g')

    port_val_credit_total = (port_val_credit_a +
                             port_val_credit_b +
                             port_val_credit_c +
                             port_val_credit_d +
                             port_val_credit_e +
                             port_val_credit_f +
                             port_val_credit_g)

    port_val_credit_a = port_val_credit_a / port_val_credit_total
    port_val_credit_b = port_val_credit_b / port_val_credit_total
    port_val_credit_c = port_val_credit_c / port_val_credit_total
    port_val_credit_d = port_val_credit_d / port_val_credit_total
    port_val_credit_e = port_val_credit_e / port_val_credit_total
    port_val_credit_f = port_val_credit_f / port_val_credit_total
    port_val_credit_g = port_val_credit_g / port_val_credit_total

    #Credit Active Weights
    act_val_credit_a = port_val_credit_a - bm_val_credit_a
    act_val_credit_b = port_val_credit_b - bm_val_credit_b
    act_val_credit_c = port_val_credit_c - bm_val_credit_c
    act_val_credit_d = port_val_credit_d - bm_val_credit_d
    act_val_credit_e = port_val_credit_e - bm_val_credit_e
    act_val_credit_f = port_val_credit_f - bm_val_credit_f
    act_val_credit_g = port_val_credit_g - bm_val_credit_g

    #Benchmark Purpose Allocation
    bm_val_purpose = pd.DataFrame(index=val_ret_index, columns=val_loans_ret.columns)
    bm_val_purpose = bm_val_purpose.where(val_loans_ret.isnull(), val_loans.loc['purpose'], axis=1)

    bm_val_car = bm_val_car = bm_val_purpose.eq('car').sum(axis=1).rename('bm_car')
    bm_val_credit_card = bm_val_purpose.eq('credit_card').sum(axis=1).rename('bm_credit_card')
    bm_val_debt_consolidation = bm_val_purpose.eq('debt_consolidation').sum(axis=1).rename('bm_debt_consolidation')
    bm_val_educational = bm_val_purpose.eq('educational').sum(axis=1).rename('bm_educational')
    bm_val_home_improvement = bm_val_purpose.eq('home_improvement').sum(axis=1).rename('bm_home_improvement')
    bm_val_house = bm_val_purpose.eq('house').sum(axis=1).rename('bm_house')
    bm_val_major_purchase = bm_val_purpose.eq('major_purchase').sum(axis=1).rename('bm_major_purchase')
    bm_val_medical = bm_val_purpose.eq('medical').sum(axis=1).rename('bm_medical')
    bm_val_moving = bm_val_purpose.eq('moving').sum(axis=1).rename('bm_moving')
    bm_val_other = bm_val_purpose.eq('other').sum(axis=1).rename('bm_other')
    bm_val_renewable_energy = bm_val_purpose.eq('renewable_energy').sum(axis=1).rename('bm_renewable_energy')
    bm_val_small_business = bm_val_purpose.eq('small_business').sum(axis=1).rename('bm_small_business')
    bm_val_vacation = bm_val_purpose.eq('vacation').sum(axis=1).rename('bm_vacation')
    bm_val_wedding = bm_val_purpose.eq('wedding').sum(axis=1).rename('bm_wedding')

    bm_val_purpose_total = (bm_val_car +
                            bm_val_credit_card +
                            bm_val_debt_consolidation +
                            bm_val_educational +
                            bm_val_home_improvement +
                            bm_val_house +
                            bm_val_major_purchase +
                            bm_val_medical +
                            bm_val_moving +
                            bm_val_other +
                            bm_val_renewable_energy +
                            bm_val_small_business +
                            bm_val_vacation +
                            bm_val_wedding)

    bm_val_car = bm_val_car / bm_val_purpose_total
    bm_val_credit_card = bm_val_credit_card / bm_val_purpose_total
    bm_val_debt_consolidation = bm_val_debt_consolidation / bm_val_purpose_total
    bm_val_educational = bm_val_educational / bm_val_purpose_total
    bm_val_home_improvement = bm_val_home_improvement / bm_val_purpose_total
    bm_val_house = bm_val_house / bm_val_purpose_total
    bm_val_major_purchase = bm_val_major_purchase / bm_val_purpose_total
    bm_val_medical = bm_val_medical / bm_val_purpose_total
    bm_val_moving = bm_val_moving / bm_val_purpose_total
    bm_val_other = bm_val_other / bm_val_purpose_total
    bm_val_renewable_energy = bm_val_renewable_energy / bm_val_purpose_total
    bm_val_small_business = bm_val_small_business / bm_val_purpose_total
    bm_val_vacation = bm_val_vacation / bm_val_purpose_total
    bm_val_wedding = bm_val_wedding / bm_val_purpose_total

    #Portfolio Purpose Allocation
    port_val_purpose = pd.DataFrame(index=val_ret_index, columns=port_val_loans_ret.columns)
    port_val_purpose = port_val_purpose.where(port_val_loans_ret.isnull(), port_val_loans.loc['purpose'], axis=1)

    port_val_car = port_val_car = port_val_purpose.eq('car').sum(axis=1).rename('port_car')
    port_val_credit_card = port_val_purpose.eq('credit_card').sum(axis=1).rename('port_credit_card')
    port_val_debt_consolidation = port_val_purpose.eq('debt_consolidation').sum(axis=1).rename(
        'port_debt_consolidation')
    port_val_educational = port_val_purpose.eq('educational').sum(axis=1).rename('port_educational')
    port_val_home_improvement = port_val_purpose.eq('home_improvement').sum(axis=1).rename('port_home_improvement')
    port_val_house = port_val_purpose.eq('house').sum(axis=1).rename('port_house')
    port_val_major_purchase = port_val_purpose.eq('major_purchase').sum(axis=1).rename('port_major_purchase')
    port_val_medical = port_val_purpose.eq('medical').sum(axis=1).rename('port_medical')
    port_val_moving = port_val_purpose.eq('moving').sum(axis=1).rename('port_moving')
    port_val_other = port_val_purpose.eq('other').sum(axis=1).rename('port_other')
    port_val_renewable_energy = port_val_purpose.eq('renewable_energy').sum(axis=1).rename('port_renewable_energy')
    port_val_small_business = port_val_purpose.eq('small_business').sum(axis=1).rename('port_small_business')
    port_val_vacation = port_val_purpose.eq('vacation').sum(axis=1).rename('port_vacation')
    port_val_wedding = port_val_purpose.eq('wedding').sum(axis=1).rename('port_wedding')

    port_val_purpose_total = (port_val_car +
                              port_val_credit_card +
                              port_val_debt_consolidation +
                              port_val_educational +
                              port_val_home_improvement +
                              port_val_house +
                              port_val_major_purchase +
                              port_val_medical +
                              port_val_moving +
                              port_val_other +
                              port_val_renewable_energy +
                              port_val_small_business +
                              port_val_vacation +
                              port_val_wedding)

    port_val_car = port_val_car / port_val_purpose_total
    port_val_credit_card = port_val_credit_card / port_val_purpose_total
    port_val_debt_consolidation = port_val_debt_consolidation / port_val_purpose_total
    port_val_educational = port_val_educational / port_val_purpose_total
    port_val_home_improvement = port_val_home_improvement / port_val_purpose_total
    port_val_house = port_val_house / port_val_purpose_total
    port_val_major_purchase = port_val_major_purchase / port_val_purpose_total
    port_val_medical = port_val_medical / port_val_purpose_total
    port_val_moving = port_val_moving / port_val_purpose_total
    port_val_other = port_val_other / port_val_purpose_total
    port_val_renewable_energy = port_val_renewable_energy / port_val_purpose_total
    port_val_small_business = port_val_small_business / port_val_purpose_total
    port_val_vacation = port_val_vacation / port_val_purpose_total
    port_val_wedding = port_val_wedding / port_val_purpose_total

    #Purpose Active Weights
    act_val_car = port_val_car - bm_val_car
    act_val_credit_card = port_val_credit_card - bm_val_credit_card
    act_val_debt_consolidation = port_val_debt_consolidation - bm_val_debt_consolidation
    act_val_educational = port_val_educational - bm_val_educational
    act_val_home_improvement = port_val_home_improvement - bm_val_home_improvement
    act_val_house = port_val_house - bm_val_house
    act_val_major_purchase = port_val_major_purchase - bm_val_major_purchase
    act_val_medical = port_val_medical - bm_val_medical
    act_val_moving = port_val_moving - bm_val_moving
    act_val_other = port_val_other - bm_val_other
    act_val_renewable_energy = port_val_renewable_energy - bm_val_renewable_energy
    act_val_small_business = port_val_small_business - bm_val_small_business
    act_val_vacation = port_val_vacation - bm_val_vacation
    act_val_wedding = port_val_wedding - bm_val_wedding

    # Charge-Off Rates
    # Total
    val_charge_off = val_loans_charge_off.eq(1).sum(axis=1).rename('val_charge_off').sum(axis=0)
    port_val_charge_off = port_val_loans_charge_off.eq(1).sum(axis=1).rename('port_charge_off').sum(axis=0)

    val_charge_off = val_charge_off / val_total_plt
    port_val_charge_off = port_val_charge_off / val_total_plt

    # Monthly Default Rate
    val_charge_off_rolling = val_loans_charge_off.eq(1).sum(axis=1).rename('val_charge_off_rolling') / bm_val_n_loans
    port_val_charge_off_rolling = port_val_loans_charge_off.eq(1).sum(axis=1).rename(
        'port_charge_off_rolling') / port_val_n_loans

    # Pre-Payment Rates
    # Total
    val_pre_pay = val_loans_pre_pay.eq(1).sum(axis=1).rename('val_pre_pay').sum(axis=0)
    port_val_pre_pay = port_val_loans_pre_pay.eq(1).sum(axis=1).rename('port_pre_pay').sum(axis=0)

    val_pre_pay = val_pre_pay / val_total_plt
    port_val_pre_pay = port_val_pre_pay / val_total_plt
    # Monthly Pre-Payment Rate
    val_pre_pay_rolling = val_loans_pre_pay.eq(1).sum(axis=1).rename('val_pre_pay_rolling') / bm_val_n_loans
    port_val_pre_pay_rolling = port_val_loans_pre_pay.eq(1).sum(axis=1).rename(
        'port_pre_pay_rolling') / port_val_n_loans

    # Benchmark total number of loans by credit rating
    bm_val_a = (val_loans.loc['grade'] == 'A').sum()
    bm_val_b = (val_loans.loc['grade'] == 'B').sum()
    bm_val_c = (val_loans.loc['grade'] == 'C').sum()
    bm_val_d = (val_loans.loc['grade'] == 'D').sum()
    bm_val_e = (val_loans.loc['grade'] == 'E').sum()
    bm_val_f = (val_loans.loc['grade'] == 'F').sum()
    bm_val_g = (val_loans.loc['grade'] == 'G').sum()
    # Benchmark total number of defaults by credit rating
    bm_val_a_co = ((val_loans.loc['grade'] == 'A') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_b_co = ((val_loans.loc['grade'] == 'B') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_c_co = ((val_loans.loc['grade'] == 'C') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_d_co = ((val_loans.loc['grade'] == 'D') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_e_co = ((val_loans.loc['grade'] == 'E') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_f_co = ((val_loans.loc['grade'] == 'F') & (val_loans.loc['charged_off'] == 1)).sum()
    bm_val_g_co = ((val_loans.loc['grade'] == 'G') & (val_loans.loc['charged_off'] == 1)).sum()
    # Benchmark default rates by credit rating
    bm_val_a_default = bm_val_a_co / bm_val_a
    bm_val_b_default = bm_val_b_co / bm_val_b
    bm_val_c_default = bm_val_c_co / bm_val_c
    bm_val_d_default = bm_val_d_co / bm_val_d
    bm_val_e_default = bm_val_e_co / bm_val_e
    bm_val_f_default = bm_val_f_co / bm_val_f
    bm_val_g_default = bm_val_g_co / bm_val_g
    # Portfolio total number of loans by credit rating
    port_val_a = (port_val_loans.loc['grade'] == 'A').sum()
    port_val_b = (port_val_loans.loc['grade'] == 'B').sum()
    port_val_c = (port_val_loans.loc['grade'] == 'C').sum()
    port_val_d = (port_val_loans.loc['grade'] == 'D').sum()
    port_val_e = (port_val_loans.loc['grade'] == 'E').sum()
    port_val_f = (port_val_loans.loc['grade'] == 'F').sum()
    port_val_g = (port_val_loans.loc['grade'] == 'G').sum()
    # Portfolio total number of defaults by credit rating
    port_val_a_co = ((port_val_loans.loc['grade'] == 'A') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_b_co = ((port_val_loans.loc['grade'] == 'B') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_c_co = ((port_val_loans.loc['grade'] == 'C') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_d_co = ((port_val_loans.loc['grade'] == 'D') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_e_co = ((port_val_loans.loc['grade'] == 'E') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_f_co = ((port_val_loans.loc['grade'] == 'F') & (port_val_loans.loc['charged_off'] == 1)).sum()
    port_val_g_co = ((port_val_loans.loc['grade'] == 'G') & (port_val_loans.loc['charged_off'] == 1)).sum()
    # Portfolio default rates by credit rating
    port_val_a_default = port_val_a_co / port_val_a
    port_val_b_default = port_val_b_co / port_val_b
    port_val_c_default = port_val_c_co / port_val_c
    port_val_d_default = port_val_d_co / port_val_d
    port_val_e_default = port_val_e_co / port_val_e
    port_val_f_default = port_val_f_co / port_val_f
    port_val_g_default = port_val_g_co / port_val_g

    #%%Export all Data to Excel Spreadsheet
    with pd.ExcelWriter(folder_path + "\\" + "RESULTS.xlsx") as writer:
        stats.to_excel(writer, sheet_name=val_or_test_label +"_STATS", index=True)
        y_val_pred.to_excel(writer, sheet_name=val_or_test_label + "_PREDICTIONS", index=True)
        y_val_score.to_excel(writer, sheet_name=val_or_test_label + "_SCORES", index=True)
        #Return Streams
        grid_results.to_excel(writer, sheet_name="GRID_RESULTS", index=False)
#%%End of data pulls and calculations
    execute_time = 'Execution Time: ' + "{:.2f}".format(((time.time() - start_time) / 60)) + ' Minutes'
#%%Report Output
    #%%Page 1 - Dataset Stats
    pdf = PdfPages(folder_path + "\\" + val_or_test_label + ' Results.pdf')

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 1', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Train / Validation / Test / Total # of loans
    labels = ['Train', 'Validation', 'Test']
    count = [train_total_plt, val_total_plt, test_total_plt]
    ax[0,1].title.set_text('Observations')
    count = ax[0,1].bar(x=labels, height=count, color=mblue)
    ax[0,1].yaxis.set_major_formatter(commas)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax[0,1].annotate('{:,}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 1 points vertical offset
                        textcoords="offset points",
                        fontsize='small',
                        ha='center', va='bottom')
    autolabel(count)

    #Number of loans in portfolios/benchmarks
    ax[1,0].title.set_text('# of Loans')
    ax[1,0].plot(bm_val_n_loans, label='Benchmark', color=mred)
    ax[1,0].plot(port_val_n_loans, label='Portfolio', color=mblue)
    ax[1,0].grid(linestyle='--')
    ax[1,0].tick_params(direction='inout', length=8)
    ax[1,0].yaxis.set_major_formatter(commas)
    ax[1,0].legend()

    #Train / Validation / Test / Total Classes
    fully_paid = (train_fully_paid_plt, val_fully_paid_plt, test_fully_paid_plt)
    charged_off = (train_charge_off_plt, val_charge_off_plt, test_charge_off_plt)
    N = 3
    ind = np.arange(N)
    ax[1,1].title.set_text('Classes')
    ax[1,1].set_ylabel('% of Dataset')
    paid = ax[1,1].bar(x=ind, height=fully_paid, color=mblue)
    charge = ax[1,1].bar(x=ind, height=charged_off, bottom=fully_paid, color=mred)
    ax[1,1].legend(loc='lower left', labels=classes)
    ax[1,1].set_xticks(ind)
    ax[1,1].set_xticklabels(labels)
    ax[1,1].yaxis.set_major_formatter(percent_0)

    for r1, r2 in zip(paid, charge):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax[1,1].text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:0.0%}'.format(h1),
                 ha="center", va="center",
                 fontsize='small')
        ax[1,1].text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:0.0%}'.format(h2),
                 ha="center", va="center",
                 color='white', fontsize='small')

    #Analysis Details
    ax[0,0].title.set_text('Analysis Details')
    ax[0,0].axis('off')
    fig.text(0.05, 0.86, timestamp_label, horizontalalignment='left', verticalalignment='center')
    fig.text(0.05, 0.82, execute_time, horizontalalignment='left', verticalalignment='center')
    fig.text(0.05, 0.78, val_date_range, horizontalalignment='left', verticalalignment='center')
    fig.text(0.05, 0.74, thresh_label, horizontalalignment='left', verticalalignment='center')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 2 - Classifier Results
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 2', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Confusion Matrix
    ax[0,0].title.set_text('Confusion Matrix')
    ax[0,0].matshow(cm_val, cmap='Blues', aspect='auto')
    ax[0,0].set_xticklabels([''] + classes)
    ax[0,0].set_yticklabels([''] + classes)
    ax[0,0].xaxis.set_ticks_position('bottom')
    ax[0,0].set_xlabel('Predicted')
    ax[0,0].set_ylabel('True')
    ax[0,0].grid(False)
    for (i, j), z in np.ndenumerate(cm_val):
        ax[0,0].text(j, i, '{:,}'.format(z), ha='center', va='top')
    for (i, j), z in np.ndenumerate(cmn_val):
        ax[0,0].text(j, i, '{:0.2%}'.format(z), ha='center', va='bottom')

    #AUROC
    fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y, y_val_score)
    optimal_thresh_roc_idx = np.argmax(tpr_val - fpr_val)
    optimal_thresh_roc = thresholds_val[optimal_thresh_roc_idx]
    print("Optimal ROC Threshold: ", optimal_thresh_roc)
    ax[1,1].title.set_text('ROC Curve')
    ax[1,1].plot(fpr_val, tpr_val, label='AUC: %0.2f' % roc_auc_val, color=color1)
    ax[1,1].plot([0, 1], [0, 1], linestyle='--', color=color_random, label='Random Classifier')
    ax[1,1].axvline(optimal_thresh_roc, ymin=0, ymax=1, color='green', linestyle='--', label='Optimal Threshold: %0.2f' % optimal_thresh_roc)
    ax[1,1].set_xlabel('False Positive Rate')
    ax[1,1].set_ylabel('True Positive Rate')
    ax[1,1].set_xlim([0.0, 1.0])
    ax[1,1].set_ylim([0.0, 1.05])
    ax[1,1].grid(linestyle='--')
    ax[1,1].tick_params(direction='inout', length=8)
    ax[1,1].legend(loc='lower right')
    ax[1,1].margins(x=0)
    ax[1,1].yaxis.set_major_formatter(percent_0)
    ax[1,1].xaxis.set_major_formatter(percent_0)

    #Precision / Recall as function of Threshold
    ax[1,0].title.set_text('Precision & Recall f(threshold)')
    p, r, threshold1 = metrics.precision_recall_curve(y, y_val_score)
    def plot_precision_recall_vs_threshold(precisions, recalls, threshold1):
        ax[1,0].plot(threshold1, precisions[:-1], color=color1, label="Precision")
        ax[1,0].plot(threshold1, recalls[:-1], color=color2, label="Recall")
        ax[1,0].axvline(threshold, ymin=0, ymax=1, color=color_random, linestyle='--', label='Current Threshold: %0.2f' % threshold)
        ax[1,0].set_ylabel("Score")
        ax[1,0].set_xlabel("Decision Threshold")
        ax[1,0].yaxis.set_major_formatter(percent_0)
        ax[1,0].xaxis.set_major_formatter(percent_0)
        ax[1,0].grid(linestyle='--')
        ax[1,0].margins(x=0)
        ax[1,0].legend(loc='best')
    plot_precision_recall_vs_threshold(p,r,threshold1)

    #Precision Recall
    p, r, threshold2 = metrics.precision_recall_curve(y, y_val_score)
    fscore = (2 * p * r) / (p + r)
    ix = np.argmax(fscore)
    optimal_thresh_pr = threshold2[ix]
    print("Optimal PR Threshold: ", optimal_thresh_pr)
    ax[0,1].title.set_text('Precision Recall')
    ax[0,1].set_ylabel('Annualized Return')
    ax[0,1].plot(r,p, color=mblue)
    ax[0,1].axvline(optimal_thresh_pr, ymin=0, ymax=1, color='green', linestyle='--', label='Optimal Threshold: %0.2f' % optimal_thresh_pr)
    ax[0,1].grid(linestyle='--')
    ax[0,1].set_xlabel('Recall')
    ax[0,1].set_ylabel('Precision')
    ax[0,1].tick_params(direction='inout', length=8)
    ax[0,1].legend(loc='upper right')
    ax[0,1].margins(x=0)
    ax[0,1].yaxis.set_major_formatter(percent_0)
    ax[0,1].xaxis.set_major_formatter(percent_0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 3 - Default & Pre-Pay
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 3', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Monthly Charge-Off Rate
    ax[0, 0].title.set_text('Monthly Charge-Off Rate')
    ax[0, 0].plot(val_charge_off_rolling, label='Benchmark', color=mred)
    ax[0, 0].plot(port_val_charge_off_rolling, label='Portfolio', color=mblue)
    ax[0, 0].grid(linestyle='--')
    ax[0, 0].tick_params(direction='inout', length=8)
    ax[0, 0].margins(x=0)
    ax[0, 0].legend(loc='upper left')
    ax[0, 0].yaxis.set_major_formatter(percent_0)

    #Total Charge-Off & Pre-Pay
    width=0.4
    labels = ['Charge-Off', 'Pre-Pay']
    N = np.arange(len(labels))
    bm = [val_charge_off,
          val_pre_pay]

    port = [port_val_charge_off,
          port_val_pre_pay]

    ax[1,0].title.set_text('Total Charge-Off & Pre-Pay Rates')
    bm = ax[1,0].bar(x=N - width/2, height=bm, color=mred, width=width, label='Benchmark')
    port = ax[1,0].bar(x=N + width/2, height=port, color=mblue, width=width, label='Portfolio')
    ax[1,0].set_xticks(N)
    ax[1,0].set_xticklabels(labels)
    ax[1,0].yaxis.set_major_formatter(percent_0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax[1,0].annotate('{:0.2%}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 1 points vertical offset
                        textcoords="offset points",
                        fontsize='x-small',
                        ha='center', va='bottom')
    autolabel(bm)
    autolabel(port)

    #Monthly Pre-Pay Rate
    ax[0, 1].title.set_text('Monthly Pre-Pay Rate')
    ax[0, 1].plot(val_pre_pay_rolling, label='Benchmark', color=mred)
    ax[0, 1].plot(port_val_pre_pay_rolling, label='Portfolio', color=mblue)
    ax[0, 1].grid(linestyle='--')
    ax[0, 1].tick_params(direction='inout', length=8)
    ax[0, 1].margins(x=0)
    ax[0, 1].yaxis.set_major_formatter(percent_0)

    # Portfolio vs Benchmark Default rates by Credit Grade
    ax[1, 1].title.set_text('Default Rate by Credit Grade')
    ax[1, 1].set_xlabel('Portfolio Default Rate')
    ax[1, 1].set_ylabel('Benchmark Default Rate')
    # A
    ax[1, 1].scatter(port_val_a_default, bm_val_a_default, label='A', color=mred)
    ax[1, 1].text(port_val_a_default, bm_val_a_default + .03, 'A', horizontalalignment='center',
                  verticalalignment='top')
    # B
    ax[1, 1].scatter(port_val_b_default, bm_val_b_default, label='B', color=mblue)
    ax[1, 1].text(port_val_b_default, bm_val_b_default + .03, 'B', horizontalalignment='center',
                  verticalalignment='top')
    # C
    ax[1, 1].scatter(port_val_c_default, bm_val_c_default, label='C', color='green')
    ax[1, 1].text(port_val_c_default, bm_val_c_default + .03, 'C', horizontalalignment='center',
                  verticalalignment='top')
    # D
    ax[1, 1].scatter(port_val_d_default, bm_val_d_default, label='D', color=pub_bm1_color)
    ax[1, 1].text(port_val_d_default, bm_val_d_default + .03, 'D', horizontalalignment='center',
                  verticalalignment='top')
    # E
    ax[1, 1].scatter(port_val_e_default, bm_val_e_default, label='E', color=pub_bm2_color)
    ax[1, 1].text(port_val_e_default, bm_val_e_default + .03, 'E', horizontalalignment='center',
                  verticalalignment='top')
    # F
    ax[1, 1].scatter(port_val_f_default, bm_val_f_default, label='F', color=pub_bm3_color)
    ax[1, 1].text(port_val_f_default, bm_val_f_default + .03, 'F', horizontalalignment='center',
                  verticalalignment='top')
    # G
    ax[1, 1].scatter(port_val_g_default, bm_val_g_default, label='G', color=mpurple)
    ax[1, 1].text(port_val_g_default, bm_val_g_default + .03, 'G', horizontalalignment='center',
                  verticalalignment='top')
    # 45 Degree Line
    lims = [
        np.min([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()]),
        np.max([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()]),
    ]
    ax[1, 1].plot(lims, lims, linestyle='--', color=color_random)

    ax[1, 1].grid(linestyle='--')
    ax[1, 1].tick_params(direction='inout', length=8)
    ax[1, 1].xaxis.set_major_formatter(percent_0)
    ax[1, 1].yaxis.set_major_formatter(percent_0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 4 - Performance 1
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 4', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Cumulative return
    ax[0,0].title.set_text('Cumulative Return')
    ax[0,0].plot(ret_cum[benchmark], label='Benchmark', color=mred)
    ax[0,0].plot(ret_cum[portfolio], label='Portfolio', color=mblue)
    ax[0,0].grid(linestyle='--')
    ax[0,0].tick_params(direction='inout', length=8)
    ax[0,0].margins(x=0)
    ax[0,0].yaxis.set_major_formatter(percent_0)
    ax[0,0].legend(loc='upper left')

    #Cumulative Excess Return
    ax[1,0].title.set_text('Cumulative Excess Return')
    ax[1,0].plot(excess_ret_cum[portfolio], color=mblue)
    ax[1,0].axhline(0, color=color_random, linestyle='--')
    ax[1,0].grid(linestyle='--')
    ax[1,0].tick_params(direction='inout', length=8)
    ax[1,0].margins(x=0)
    ax[1,0].yaxis.set_major_formatter(percent_1)

    #Rolling Return
    ax[0,1].title.set_text('Rolling ' + format(window) + 'M' + ' Return')
    ax[0,1].set_ylabel('Annualized Return')
    ax[0,1].plot(ret_rolling[benchmark], label='Benchmark', color=mred)
    ax[0,1].plot(ret_rolling[portfolio], label='Portfolio', color=mblue)
    ax[0,1].grid(linestyle='--')
    ax[0,1].tick_params(direction='inout', length=8)
    ax[0,1].margins(x=0)
    ax[0,1].yaxis.set_major_formatter(percent_0)

    #Rolling Excess Return
    ax[1,1].title.set_text('Rolling ' + format(window) + 'M' + ' Excess Return')
    ax[1,1].set_ylabel('Annualized Return')
    ax[1,1].plot(excess_ret_rolling[portfolio], color=mblue)
    ax[1,1].axhline(0, color=color_random, linestyle='--')
    ax[1,1].grid(linestyle='--')
    ax[1,1].tick_params(direction='inout', length=8)
    ax[1,1].margins(x=0)
    ax[1,1].yaxis.set_major_formatter(percent_1)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 5 - Performance 2
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 5', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Rolling Risk
    ax[0,0].title.set_text('Rolling ' + format(window) + 'M' + ' Risk')
    ax[0,0].set_ylabel('Annualized Risk')
    ax[0,0].plot(std_rolling[benchmark], label='Benchmark', color=mred)
    ax[0,0].plot(std_rolling[portfolio], label='Portfolio', color=mblue)
    ax[0,0].grid(linestyle='--')
    ax[0,0].tick_params(direction='inout', length=8)
    ax[0,0].margins(x=0)
    ax[0,0].yaxis.set_major_formatter(percent_1)
    ax[0,0].legend(loc='upper left')

    #Rolling TE
    ax[1,0].title.set_text('Rolling ' + format(window) + 'M' + ' Tracking Error')
    ax[1,0].set_ylabel('Annualized Tracking Error')
    ax[1,0].plot(te_rolling[portfolio], color=mblue)
    ax[1,0].grid(linestyle='--')
    ax[1,0].tick_params(direction='inout', length=8)
    ax[1,0].margins(x=0)
    ax[1,0].yaxis.set_major_formatter(percent_1)

    #Cumulative DD
    ax[0,1].title.set_text('Cumulative Drawdown')
    ax[0,1].plot(drawdown[benchmark], label='Benchmark', color=mred)
    ax[0,1].plot(drawdown[portfolio], label='Portfolio', color=mblue)
    ax[0,1].grid(linestyle='--')
    ax[0,1].margins(x=0)
    ax[0,1].yaxis.set_major_formatter(percent_1)

    #Risk Return Chart
    ax[1,1].title.set_text('Risk & Return')
    ax[1,1].set_xlabel('Annualized Risk')
    ax[1,1].set_ylabel('Annualized Return')
    ax[1,1].scatter(std_incep[benchmark], ret_incep[benchmark], label='Benchmark', color=mred)
    ax[1,1].text(std_incep[benchmark] + .0025, ret_incep[benchmark] + .0025, 'Benchmark', horizontalalignment='left',
                  verticalalignment='top')
    ax[1,1].scatter(std_incep[portfolio], ret_incep[portfolio], label='Portfolio', color=mblue)
    ax[1,1].text(std_incep[portfolio] + .0025, ret_incep[portfolio] + .0025, 'Portfolio', horizontalalignment='left',
                  verticalalignment='top')
    ax[1,1].scatter(std_incep[cash], ret_incep[cash], label='Cash', color='green')
    ax[1,1].text(std_incep[cash] + .0025, ret_incep[cash] + .0025, 'Cash', horizontalalignment='left',
                  verticalalignment='top')
    ax[1,1].scatter(std_incep[pub_bm1], ret_incep[pub_bm1], label=pub_bm1, color=pub_bm1_color)
    ax[1,1].text(std_incep[pub_bm1] + .0025, ret_incep[pub_bm1] + .0025, pub_bm1, horizontalalignment='right',
                  verticalalignment='center')
    ax[1,1].scatter(std_incep[pub_bm2], ret_incep[pub_bm2], label=pub_bm2, color=pub_bm2_color)
    ax[1,1].text(std_incep[pub_bm2] + .0025, ret_incep[pub_bm2] + .0025, pub_bm2, horizontalalignment='left',
                  verticalalignment='top')
    ax[1,1].scatter(std_incep[pub_bm3], ret_incep[pub_bm3], label=pub_bm3, color=pub_bm3_color)
    ax[1,1].text(std_incep[pub_bm3] + .0025, ret_incep[pub_bm3] + .0025, pub_bm3, horizontalalignment='left',
                  verticalalignment='top')
    ax[1,1].grid(linestyle='--')
    ax[1,1].tick_params(direction='inout', length=8)
    ax[1,1].xaxis.set_major_formatter(percent_0)
    ax[1,1].yaxis.set_major_formatter(percent_0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 6 - Performance 3
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 6', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    # Rolling Sharpe
    ax[0,0].title.set_text('Rolling ' + format(window) + 'M' + ' Sharpe Ratio')
    ax[0,0].plot(sr_rolling[benchmark], label='Benchmark', color=mred)
    ax[0,0].plot(sr_rolling[portfolio], label='Portfolio', color=mblue)
    ax[0,0].grid(linestyle='--')
    ax[0,0].tick_params(direction='inout', length=8)
    ax[0,0].margins(x=0)
    ax[0,0].legend(loc='upper left')

    # Rolling Information Ratio
    ax[1,0].title.set_text('Rolling ' + format(window) + 'M' + ' Info Ratio')
    ax[1,0].plot(ir_rolling[portfolio], label='Portfolio', color=mblue)
    ax[1,0].axhline(0, color=color_random, linestyle='--')
    ax[1,0].grid(linestyle='--')
    ax[1,0].tick_params(direction='inout', length=8)
    ax[1,0].margins(x=0)

    # Rolling Alpha
    ax[0,1].title.set_text('Rolling ' + format(window) + 'M' + ' Alpha')
    ax[0,1].plot(alpha_rolling[portfolio], color=mblue)
    ax[0,1].grid(linestyle='--')
    ax[0,1].tick_params(direction='inout', length=8)
    ax[0,1].margins(x=0)
    ax[0,1].yaxis.set_major_formatter(percent_1)

    # Rolling Beta
    ax[1,1].title.set_text('Rolling ' + format(window) + 'M' + ' Beta')
    ax[1,1].plot(beta_rolling[portfolio], color=mblue)
    ax[1,1].grid(linestyle='--')
    ax[1,1].tick_params(direction='inout', length=8)
    ax[1,1].margins(x=0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 7 - Performance 4
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 7', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    width = 0.4
    #Trailing Return vs Benchmark
    labels = ['1Yr', '3Yr', '5Yr', '10Yr', 'Incep']
    N = np.arange(len(labels))
    bm = [ret_1yr_trail[benchmark],
          ret_3yr_trail[benchmark],
          ret_5yr_trail[benchmark],
          ret_10yr_trail[benchmark],
          ret_incep[benchmark]]
    port = [ret_1yr_trail[portfolio],
          ret_3yr_trail[portfolio],
          ret_5yr_trail[portfolio],
          ret_10yr_trail[portfolio],
          ret_incep[portfolio]]
    ax[0,0].title.set_text('Trailing Return')
    bm = ax[0,0].bar(x=N - width/2, height=bm, color=mred, width=width, label='Benchmark')
    port = ax[0,0].bar(x=N + width/2, height=port, color=mblue, width=width, label='Portfolio')
    ax[0,0].set_ylabel('Annualized Return')
    ax[0,0].set_xticks(N)
    ax[0,0].set_xticklabels(labels)
    ax[0,0].yaxis.set_major_formatter(percent_1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax[0,0].annotate('{:0.2%}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        fontsize='x-small',
                        ha='center', va='bottom')
    autolabel(bm)
    autolabel(port)

    #Trailing Sharpe Ratio vs Benchmark
    labels = ['3Yr', '5Yr', '10Yr', 'Incep']
    N = np.arange(len(labels))
    bm = [sr_3yr_trail[benchmark],
          sr_5yr_trail[benchmark],
          sr_10yr_trail[benchmark],
          sr_incep[benchmark]]
    port = [sr_3yr_trail[portfolio],
          sr_5yr_trail[portfolio],
          sr_10yr_trail[portfolio],
          sr_incep[portfolio]]
    ax[1,0].title.set_text('Trailing Sharpe Ratio')
    bm = ax[1,0].bar(x=N - width/2, height=bm, color=mred, width=width, label='Benchmark')
    port = ax[1,0].bar(x=N + width/2, height=port, color=mblue, width=width, label='Portfolio')
    ax[1,0].set_ylabel('Sharpe Ratio')
    ax[1,0].set_xticks(N)
    ax[1,0].set_xticklabels(labels)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax[1,0].annotate('{:0.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 1 points vertical offset
                        textcoords="offset points",
                        fontsize='x-small',
                        ha='center', va='bottom')
    autolabel(bm)
    autolabel(port)

    #Correlation Matrix
    labels = corr_incep.rename(columns={'Benchmark': 'bm_val','Portfolio':'port_val'}, index={'Benchmark':'bm_val', 'Portfolio':'port_val'})
    labels = labels.columns
    ax[0,1].title.set_text('Correlation Matrix (Since Inception)')
    ax[0,1].matshow(corr_incep, cmap='Blues', aspect='auto')
    ax[0,1].set_xticks(np.arange(0, corr_incep.shape[0], corr_incep.shape[0] * 1.0 / len(labels)))
    ax[0,1].set_yticks(np.arange(0, corr_incep.shape[1], corr_incep.shape[1] * 1.0 / len(labels)))
    ax[0,1].set_xticklabels('')
    ax[0,1].set_yticklabels(labels)
    for (i, j), z in np.ndenumerate(corr_incep):
        ax[0,1].text(j, i, '{:.2f}'.format(z), ha='center', va='center')

    #Since Inception Stats Table
    ax[1,1].title.set_text('Since Inception vs Benchmark')
    plot_table = stats[[portfolio, benchmark]]
    plot_table = plot_table.loc(axis=0)[
        'incep_info',  # Float
        'incep_excess',  # Percent
        'incep_alpha',  # Percent
        'incep_te',  # Percent
        'incep_beta',  # Float
        'incep_r2',  # Percent
        'max_drawdown',  # Percent
        'min_return',  # Percent
        'max_return',  # Percent
        'mean_return'  # Percent
    ]
    plot_table.update(plot_table.loc(axis=0)[['incep_beta',
                                              'incep_r2',
                                              'incep_info']].applymap('{:,.2f}'.format))

    plot_table.update(plot_table.loc(axis=0)[['incep_excess',
                                              'incep_alpha',
                                              'incep_te',
                                              'max_drawdown',
                                              'min_return',
                                              'max_return',
                                              'mean_return']].applymap('{:,.2%}'.format))

    columns = ('Portfolio', 'Benchmark')
    rows = ('Information Ratio',
            'Excess Return',
            'Alpha',
            'Tracking Error',
            'Beta',
            'Correlation',
            'Max Drawdown',
            'Worst Month',
            'Best Month',
            'Mean Month')
    cell_text = []
    for row in range(len(plot_table)):
        cell_text.append(plot_table.iloc[row])

    ax[1,1].table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center', bbox=[0,0,1,1])
    ax[1,1].axis('tight')
    ax[1,1].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 8 - Characteristics 1
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 8', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    #Interest Rate
    ax[0,0].title.set_text('Weighted Average Interest Rate')
    ax[0,0].plot(bm_val_int_rate, label='Benchmark', color=mred)
    ax[0,0].plot(port_val_int_rate, label='Portfolio', color=mblue)
    ax[0,0].grid(linestyle='--')
    ax[0,0].tick_params(direction='inout', length=8)
    ax[0,0].margins(x=0)
    ax[0,0].yaxis.set_major_formatter(percent_1)
    ax[0,0].legend(loc='upper left')

    #Debt to Income
    ax[1,0].title.set_text('Weighted Average Debt to Income')
    ax[1,0].plot(bm_val_dti, label='Benchmark', color=mred)
    ax[1,0].plot(port_val_dti, label='Portfolio', color=mblue)
    ax[1,0].grid(linestyle='--')
    ax[1,0].tick_params(direction='inout', length=8)
    ax[1,0].margins(x=0)
    ax[1,0].yaxis.set_major_formatter(percent_1)

    #FICO Score
    ax[0,1].title.set_text('Weighted Average FICO Score')
    ax[0,1].plot(bm_val_fico, label='Benchmark', color=mred)
    ax[0,1].plot(port_val_fico, label='Portfolio', color=mblue)
    ax[0,1].grid(linestyle='--')
    ax[0,1].tick_params(direction='inout', length=8)
    ax[0,1].margins(x=0)

    #Loan Amount
    ax[1,1].title.set_text('Weighted Average Loan Amount')
    ax[1,1].plot(bm_val_loan_amnt, label='Benchmark', color=mred)
    ax[1,1].plot(port_val_loan_amnt, label='Portfolio', color=mblue)
    ax[1,1].grid(linestyle='--')
    ax[1,1].tick_params(direction='inout', length=8)
    ax[1,1].yaxis.set_major_formatter(dollars)
    ax[1,1].margins(x=0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 9 -Characteristics 2
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 9', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    # Average Credit Rating
    y_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    y_axis = np.arange(1, 7, 1)

    ax[0, 0].title.set_text('Weighted Average Grade')
    ax[0, 0].plot(bm_val_credit_avg, label='Benchmark', color=mred)
    ax[0, 0].plot(port_val_credit_avg, label='Portfolio', color=mblue)
    ax[0, 0].grid(linestyle='--')
    ax[0, 0].tick_params(direction='inout', length=8)
    ax[0, 0].margins(x=0)
    ax[0, 0].set_yticks(y_axis)
    ax[0, 0].set_yticklabels(y_labels)
    ax[0, 0].legend(loc='upper left')

    # Active Credit Weights
    ax[1, 0].title.set_text('Active Grade Weight')
    ax[1, 0].set_ylabel('Portfolio Less Benchmark Weight')
    ax[1, 0].plot(act_val_credit_a, label='A', color='#008AC9')
    ax[1, 0].plot(act_val_credit_b, label='B', color='#2B115A')
    ax[1, 0].plot(act_val_credit_c, label='C', color='#f11a21')
    ax[1, 0].plot(act_val_credit_d, label='D', color='#20b2aa')
    ax[1, 0].plot(act_val_credit_e, label='E', color='#fa8072')
    ax[1, 0].plot(act_val_credit_f, label='F', color='#bfeebf')
    ax[1, 0].plot(act_val_credit_g, label='G', color='#407294')
    ax[1, 0].grid(linestyle='--')
    ax[1, 0].legend(loc='upper left')
    ax[1, 0].tick_params(direction='inout', length=8)
    ax[1, 0].margins(x=0)
    ax[1, 0].yaxis.set_major_formatter(percent_0)

    # Average Term
    ax[0, 1].title.set_text('Weighted Average Term')
    ax[0, 1].set_ylabel('Months')
    ax[0, 1].plot(bm_val_term_avg, label='Benchmark', color=mred)
    ax[0, 1].plot(port_val_term_avg, label='Portfolio', color=mblue)
    ax[0, 1].grid(linestyle='--')
    ax[0, 1].tick_params(direction='inout', length=8)
    ax[0, 1].set_ylim(bottom=30)
    ax[0, 1].margins(x=0)

    # Purpose Active Weight
    ax[1, 1].title.set_text('Active Purpose Weight')
    ax[1, 1].set_ylabel('Portfolio Less Benchmark Weight')
    ax[1, 1].plot(act_val_car, label='Car', color='#008AC9')
    ax[1, 1].plot(act_val_credit_card, label='Credit Card', color='#2B115A')
    ax[1, 1].plot(act_val_debt_consolidation, label='Debt Consol', color='#f11a21')
    ax[1, 1].plot(act_val_educational, label='Edu', color='#20b2aa')
    ax[1, 1].plot(act_val_home_improvement, label='Home Improv', color='#fa8072')
    ax[1, 1].plot(act_val_house, label='House', color='#bfeebf')
    ax[1, 1].plot(act_val_major_purchase, label='Major Pur', color='#407294')
    ax[1, 1].plot(act_val_medical, label='Medical', color='#00b8ff')
    ax[1, 1].plot(act_val_moving, label='Moving', color='#785aa8')
    ax[1, 1].plot(act_val_other, label='Other', color='#daa520')
    ax[1, 1].plot(act_val_renewable_energy, label='Renew Energy', color='#002f8a')
    ax[1, 1].plot(act_val_small_business, label='Small Biz', color='#ffe700')
    ax[1, 1].plot(act_val_vacation, label='Vacation', color='#794044')
    ax[1, 1].plot(act_val_wedding, label='Wedding', color='#133337')
    ax[1, 1].grid(linestyle='--')
    ax[1, 1].legend(loc='upper left')
    ax[1, 1].tick_params(direction='inout', length=8)
    ax[1, 1].yaxis.set_major_formatter(percent_0)
    ax[1, 1].margins(x=0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')

    #%%Page 10 - Characteristics 3
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(11, 8.5)
    fig.text(0.99, 0.01, author, color=color_random, horizontalalignment='right')
    fig.text(0.5, 0.01, 'Page 10', horizontalalignment='right')
    fig.suptitle(t= model_name + ': ' + val_or_test_label, fontsize='x-large')

    credit_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    purpose_labels = ['Car',
                      'Credit Card'
                      'Debt Consol',
                      'Edu',
                      'Home Improv',
                      'House',
                      'Major Pur',
                      'Medical',
                      'Moving',
                      'Other',
                      'Renew Energy',
                      'Small Biz',
                      'Vacation',
                      'Wedding']

    #Portfolio Credit Allocation
    ax[0, 0].title.set_text('Portfolio Grade')
    ax[0, 0].stackplot(val_ret_index, port_val_credit_a,
                       port_val_credit_b,
                       port_val_credit_c,
                       port_val_credit_d,
                       port_val_credit_e,
                       port_val_credit_f,
                       port_val_credit_g,
                       baseline='zero', labels=credit_labels)
    ax[0, 0].margins(x=0)
    ax[0, 0].yaxis.set_major_formatter(percent_0)
    ax[0, 0].set_ylim(top=1)
    ax[0, 0].legend(loc='upper left')

    #Portfolio Purpose Allocation
    ax[1, 0].title.set_text('Portfolio Purpose')
    ax[1, 0].stackplot(val_ret_index, port_val_car,
                       port_val_credit_card,
                       port_val_debt_consolidation,
                       port_val_educational,
                       port_val_home_improvement,
                       port_val_house,
                       port_val_major_purchase,
                       port_val_medical,
                       port_val_moving,
                       port_val_other,
                       port_val_renewable_energy,
                       port_val_small_business,
                       port_val_vacation,
                       port_val_wedding,
                       baseline='zero', labels=purpose_labels)
    ax[1, 0].tick_params(direction='inout', length=8)
    ax[1, 0].margins(x=0)
    ax[1, 0].set_ylim(top=1)
    ax[1, 0].legend(loc='upper left')
    ax[1, 0].yaxis.set_major_formatter(percent_0)

    #Benchmark Credit allocation
    ax[0, 1].title.set_text('Benchmark Grade')
    ax[0, 1].stackplot(val_ret_index, bm_val_credit_a,
                       bm_val_credit_b,
                       bm_val_credit_c,
                       bm_val_credit_d,
                       bm_val_credit_e,
                       bm_val_credit_f,
                       bm_val_credit_g,
                       baseline='zero', labels=credit_labels)
    ax[0, 1].margins(x=0)
    ax[0, 1].set_ylim(top=1)
    ax[0, 1].yaxis.set_major_formatter(percent_0)
    ax[0, 1].legend(loc='upper left')

    #Benchmark Purpose Allocation
    ax[1, 1].title.set_text('Benchmark Purpose')
    ax[1, 1].stackplot(val_ret_index, bm_val_car,
                       bm_val_credit_card,
                       bm_val_debt_consolidation,
                       bm_val_educational,
                       bm_val_home_improvement,
                       bm_val_house,
                       bm_val_major_purchase,
                       bm_val_medical,
                       bm_val_moving,
                       bm_val_other,
                       bm_val_renewable_energy,
                       bm_val_small_business,
                       bm_val_vacation,
                       bm_val_wedding,
                       baseline='zero', labels=purpose_labels)
    ax[1, 1].tick_params(direction='inout', length=8)
    ax[1, 1].margins(x=0)
    ax[1, 1].set_ylim(top=1)
    ax[1, 1].legend(loc='upper left')
    ax[1, 1].yaxis.set_major_formatter(percent_0)

    fig.subplots_adjust(top=0.9, bottom=0.10)
    plt.savefig(pdf, format='pdf')
    pdf.close()
#%%
    #Total Execution Time
    print("--- Total Time to Execute: %s Seconds ---" % (time.time() - start_time),file=open(folder_path + "\\" + "MODEL_OUTPUT.txt", "a"))
    print("--- Total Time to Execute: %s Seconds ---" % (time.time() - start_time))

#%%Generate Model Results
model_results(model=model,
              model_name=model_name,
              X=X_val,
              y=y_val,
              val_or_test='val',
              threshold=0.5,
              window=36)
#%%Send email confirming Script ran correctly
port = 465  # For SSL
smtp_server = "XXXX"
sender_email = "XXXX"  # Enter your address
receiver_email = "XXXX"  # Enter receiver address
password =open(r"XXXXX").read()
message = """\
Subject: CL_LOGREG Code Executed

This message is sent from MEGA DESK."""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
