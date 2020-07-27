#Author: Max H. Rowland
#Email: maxh.rowland@gmail.com
#Script uses a random forest classifier to predict loan defaults within the lending Club dataset
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
#Model Specific Packages
from sklearn.ensemble import RandomForestClassifier
#%%Global model output settings
#######################################################################################################################
#######################################################################################################################
model_name = 'Random Forest' #Name used in chart titles
save_path = r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\CL_RFC' #Directory for saving model output
scorer = 'roc_auc' #Scoring metric for Grid Search
start_time = time.time() #start time for execution timer
#######################################################################################################################
#######################################################################################################################
#%%Load Train/Val/Test Data Files
#######################################################################################################################
#######################################################################################################################
#Training CSV files
X_train = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\TRAIN\loans_SMOTE_X_train_all.CSV')
y_train = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\TRAIN\loans_SMOTE_y_train_all.CSV')

#Validation CSV files
X_val = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\VAL\loans_IS_X_val_all.CSV').set_index('id')
y_val = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\VAL\loans_RAW_y_val_all.CSV').set_index('id')

#Test CSV files
X_test = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\TEST\loans_IS_X_test_all.CSV').set_index('id')
y_test = pd.read_csv(r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\DATA\TEST\loans_RAW_y_test_all.CSV').set_index('id')
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

#%%Random Forest Classifier
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
pipeline = Pipeline([
    ('model', RandomForestClassifier(n_estimators=50, #The number of trees in the forest - default = 100
                                     criterion='gini', #The function to measure the quality of a split - default 'gini'
                                     max_depth=10, #The maximum depth of the tree - default = None
                                     min_samples_split=2, #The minimum number of samples required to split an internal node - default = 2
                                     min_samples_leaf=1, #The minimum number of samples required to be at a leaf node - default = 1
                                     min_weight_fraction_leaf=0.0, #The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node - default = 0
                                     max_features='auto', #The number of features to consider when looking for the best split - default = 'auto'
                                     max_leaf_nodes=None, #Grow trees with max_leaf_nodes in best-first fashion - default = None
                                     min_impurity_decrease=0.0, #A node will be split if this split induces a decrease of the impurity greater than or equal to this value - default = 0
                                     bootstrap=False, #Whether bootstrap samples are used when building trees - default = True
                                     oob_score=False, #Whether to use out-of-bag samples to estimate the generalization accuracy - default = False
                                     n_jobs=8, #Number of CPUs to use - default = None
                                     random_state=1989, #The seed of random number generator - default = None
                                     verbose=0, #How often progress messages are printed - default = 0
                                     warm_start=False, #When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest - default = False
                                     class_weight=None, #Weights associated with classes in the form {class_label: weight} - default = None
                                     ccp_alpha=0, #Complexity parameter used for Minimal Cost-Complexity Pruning - default = 0
                                     max_samples=None) #If bootstrap is True, the number of samples to draw from X to train each base estimator - default = none
     )
])
param_grid = {
    #'model__n_estimators': [50,100,250],
    #'model__criterion': ['gini'],
    #'model__max_depth': [None, 5, 10],
    #model__min_samples_split': [2,5],
    #model__min_samples_leaf': [10,20],
    #'model__min_weight_fraction_leaf': [50],
    #model__max_features': ['sqrt'],
    #'model__max_leaf_nodes': [50],
    #'model__min_impurity_decrease': [50],
    #'model__min_impurity_split': [50],
    #'model__bootstrap': [50],
    #'model__oob_score': [50],
    #'model__warm_start': [50],
    #'model__class_weight': [50],
    #'model__ccp_alpha': [50],
    #'model__max_samples': [50],
}
scorers = {
    'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
    'precision_score': metrics.make_scorer(metrics.precision_score),
    'recall_score': metrics.make_scorer(metrics.recall_score),
    'accuracy_score': metrics.make_scorer(metrics.accuracy_score)
}
model = GridSearchCV(estimator=pipeline, #Model
                     param_grid=param_grid, #Search grip parameters
                     scoring=scorers, #evaluate the predictions on the test set - default = None
                     n_jobs=8, #Number of CPUs to use - default = None
                     refit=scorer, #Refit an estimator using the best found parameters on the whole dataset. - default = True
                     cv=5, #Determines the cross-validation splitting strategy
                     verbose=0, #How often progress messages are printed - default = 0
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
