#Author: Max H. Rowland
#Email: maxh.rowland@gmail.com
#Script uses a K-nearest neighbors classifier to predict loan defaults within the lending Club dataset
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
#Model Specific Packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#%%Global model output settings
#######################################################################################################################
#######################################################################################################################
model_name = 'K-Nearest Neighbors' #Name used in chart titles
save_path = r'C:\Users\mhr19\Dropbox\CODE\CONSUMER_DEBT\CL_KNN' #Directory for saving model output
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

#%%K-nearest Neighbors
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
pipeline = Pipeline([
    ('lda', LinearDiscriminantAnalysis()),
    ('model', KNeighborsClassifier(n_neighbors=5, #Number of neighbors to use by default for kneighbors queries - default = 5
                                   weights='uniform', #weight function used in prediction - default = 'uniform'
                                   algorithm='auto', #Algorithm used to compute the nearest neighbors - default = 'auto'
                                   leaf_size=30, #Leaf size passed to BallTree or KDTree - default = 30
                                   p=2, #Power parameter for the Minkowski metric - default = 2
                                   metric='minkowski', #the distance metric to use for the tree - default = 'minkowski'
                                   metric_params=None, #Additional keyword arguments for the metric function - None
                                   n_jobs=8) #The number of parallel jobs to run for neighbors search - default = None
     )
])
param_grid = {
    'lda__n_components': [12], # Number of LDA components to keep
    'model__n_neighbors': [300], # The 'k' in k-nearest neighbors
    'model__weights':['uniform'],
    'model__algorithm':['auto'],
    'model__leaf_size':[50],
    'model__p':[2],
    'model__metric':['minkowski'],
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

