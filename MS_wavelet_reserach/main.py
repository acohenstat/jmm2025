
import pandas as pd # data analysis and manipulation
import numpy as np # linear algebra
import glob # Glob moule searches all path names

import pywt # Wavelet transform in Python

from scipy.stats import kurtosis as kurt
from scipy.stats import skew
import statsmodels.api as sm


from sklearn.preprocessing import StandardScaler # z score scaling
from sklearn.decomposition import PCA # for Principal Component Analysis
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline #make pipeline
from sklearn.metrics import  accuracy_score, recall_score, precision_score, f1_score   # draws a confusion matrix

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings
import itertools

# Get a list of all the csv files
csv_filesC = glob.glob('Cancer/*.csv') # cDncer diagnosis datasets
csv_filesN = glob.glob('Normal/*.csv') # normal diagnosis datasets
wavelets = ['bior3.1', 'coif6','sym20', 'db14', 'dmey', 'coif7', 'db13', 'db15','db16', 'db12',]
# Create an empty list to hold dataframes for cancer patient
dfC = []

# Loop over csv files and append to dfC lists
for csv in csv_filesC:
    df = pd.read_csv(csv)
    dfC.append(df)
    
# Create an empty list to hold dataframes for normal patient
dfN = []
# Loop over csv files and append to dfN lists
for csv in csv_filesN:
    df = pd.read_csv(csv)
    dfN.append(df)
    

def Z_matrix(X, J):
    # Calculate N and n
        
    N = 2**J # length of windows
    n = len(X) // N # n windows

    # Divide X into n windows of window size N
    Z = np.array([X[i:i+N] for i in range(0, len(X), N)])
    return np.transpose(Z)

def WAVEDEC(Z, wavelet):
    coeffs= pywt.wavedec(Z, wavelet, 'sym', level =None , axis = 0) # Apply DWT to Intensity column of each subarray    
    # Unpack the coefficients
    cAn, *cDn = coeffs
    return cAn, cDn
def features1(coef):
    feat = np.array([np.sum(coef**2, axis = 0),                                                      # energy
                        np.mean(coef, axis =0),                                                      # mean
                        np.var(coef, ddof = 1, axis =0),                                             # variance
                        np.sqrt(np.var(coef, ddof = 1, axis =0))/np.mean(coef, axis =0),             # coeficient of variance
                        skew(coef, axis = 0, bias =False),                                           # skewness
                        kurt(coef, axis = 0, bias =False)                                            # kurtosis
                        ]).T
    return feat

def pca_comp(features):
    sc = StandardScaler() # initiate scaler function
    feature_scaled = sc.fit_transform(features) # fit and transform -> standardize the features
    pca = PCA(n_components = 0.95) # initiate PCA function
    pcs = pca.fit_transform(feature_scaled)
    return pcs

def hotelling_t1(pcs_a, pcs_d):
    # foo calculates the hotelling t-square for approx and each detail principal component coefficients
    def foo(pcs):
        mu = np.mean(pcs, axis = 0) # mean of pcs
        cov = np.cov(pcs.T) # covariance of pcs
        cov_inv = np.linalg.inv(cov) # inverse of covaraince
        t2 = np.array([np.dot(np.dot((row - mu), cov_inv), (row - mu).T) for row in pcs])
        return t2
    t2a = foo(pcs_a)
    t2d = np.array([foo(pcs_d[d]) for d in range(len(pcs_d))])
    
    return np.sqrt(np.sum(np.vstack((t2a,t2d)).T, axis =1)) 



def preprocess1(df, J, wavelet):
    t_squared =[]
    for i in range(len(df)):
        data = df[i].Intensity.iloc[:32768]
        Z = Z_matrix(data, J)
        cAn, cDn = WAVEDEC(Z, wavelet)
        
        featA = features1(cAn) # compute the features for approx
        featD = [features1(cDn[d]) for d in range(len(cDn))] 
        
        pcs_a = pca_comp(featA)
        pcs_d = [pca_comp(featD[d]) for d in range(len(featD))]
        t2 = hotelling_t1(pcs_a, pcs_d)
        t_squared.append(t2)
    return t_squared



def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    Returns a pandas df with the F1, recall, precision, and accuracy scores
    '''
    
    # Extract accuracy, precision, recall, and f1 score
    f1 = f1_score(y_test, y_test_preds)
    recall = recall_score(y_test, y_test_preds)
    precision = precision_score(y_test, y_test_preds)
    accuracy_train = accuracy_score(y_train, y_train_preds)
    accuracy_val = accuracy_score(y_val, y_val_preds)
    accuracy_test = accuracy_score(y_test, y_test_preds)
    missclassified = [(y_test != y_test_preds).sum()]
    
    # Create table of results
    table = pd.DataFrame(
        {'model': [model_name],
         'misclassified examples': missclassified,
        'precision': [precision],
        'recall': [recall],
        'F1 score': [f1],
        'accuracy_train': [accuracy_train],
        'accuracy_val': [accuracy_val],
        'accuracy_test': [accuracy_test],
        },
    )
    return table

def features2(coef, z):
    acf =  np.array([sm.tsa.acf(z[:,i], nlags=10) for i in range(z.shape[1])])                       # 10 lag feature
    feat = np.array([np.sum(coef**2, axis = 0),                                                      # energy
                        np.mean(coef, axis =0),                                                      # mean
                        np.var(coef, ddof = 1, axis =0),                                             # variance
                        np.sqrt(np.var(coef, ddof = 1, axis =0))/np.mean(coef, axis =0),             # coeficient of variance
                        skew(coef, axis = 0, bias =False),                                           # skewness
                        kurt(coef, axis = 0, bias =False),                                           # kurtosis
                        ]).T
    return np.hstack([feat,acf]) #

def preprocess2(df, J, wavelet):
    t_squared =[]
    for i in range(len(df)):
        data = df[i].Intensity.iloc[:32768]
        Z = Z_matrix(data, J)
        cAn, cDn = WAVEDEC(Z, wavelet)
        
        featA = features2(cAn, Z) # compute the features for approx
        featD = [features2(cDn[d], Z) for d in range(len(cDn))] 
        
        pcs_a = pca_comp(featA)
        pcs_d = [pca_comp(featD[d]) for d in range(len(featD))]
        t2 = hotelling_t1(pcs_a, pcs_d)
        t_squared.append(t2)
    return t_squared

def features3(coef):
    feat = np.array([np.sum(coef**2, axis = 0),                                                      # energy
                        np.mean(coef, axis =0),                                                      # mean
                        np.var(coef, ddof = 1, axis =0),                                             # variance
                        np.sqrt(np.var(coef, ddof = 1, axis =0))/np.mean(coef, axis =0),             # coeficient of variance
                        skew(coef, axis = 0, bias =False),                                           # skewness
                        kurt(coef, axis = 0, bias =False),                                           # kurtosis
                        ]).T
    return np.hstack([feat]) #

def hotelling_t3(pcs):
# foo calculates the hotelling t-square for approx and each detail principal component coefficients
    mu = np.mean(pcs, axis = 0) # mean of pcs
    cov = np.cov(pcs.T) # covariance of pcs
    cov_inv = np.linalg.inv(cov) # inverse of covaraince
    t2 = np.array([np.dot(np.dot((row - mu), cov_inv), (row - mu).T) for row in pcs])  
    return np.sqrt(t2) 
def preprocess3(df, J):
    t_squared =[]
    for i in range(len(df)):
        data = df[i].Intensity.iloc[:32768]
        Z = Z_matrix(data, J)
        feat = features3(Z) # compute the features for approx
        pcs = pca_comp(feat)
        t2 = hotelling_t3(pcs)
        t_squared.append(t2)
    return t_squared
def features4(coef):
      
    
    acf =  np.array([sm.tsa.acf(coef[:,i], nlags=10) for i in range(coef.shape[1])])                       # 10 lag feature
    feat = np.array([np.sum(coef**2, axis = 0),                                                      # energy
                        np.mean(coef, axis =0),                                                      # mean
                        np.var(coef, ddof = 1, axis =0),                                             # variance
                        np.sqrt(np.var(coef, ddof = 1, axis =0))/np.mean(coef, axis =0),             # coeficient of variance
                        skew(coef, axis = 0, bias =False),                                           # skewness
                        kurt(coef, axis = 0, bias =False),                                           # kurtosis
                        ]).T
    return np.hstack([feat,acf]) #
def preprocess4(df, J):
    t_squared =[]
    for i in range(len(df)):
        data = df[i].Intensity.iloc[:32768]
        Z = Z_matrix(data, J)
        feat = features4(Z) # compute the features for approx
        pcs = pca_comp(feat)
        t2 = hotelling_t3(pcs)
        t_squared.append(t2)
    return t_squared



from collections import Counter
# Define the preprocess functions and settings

files  = 1
print('processing_______________________________________')
while files < 100:
    # Initialize an empty list to store results
    
    preprocess_functions = [preprocess1, preprocess2]
    wavelets = ['bior3.1', 'coif6', 'sym20', 'db14', 'dmey', 'coif7', 'db13', 'db15','db16', 'db12',]
    # 'sym20', 'db14', 'dmey', 'coif7', 'db13', 'db15','db16', 'db12', 'sym19', 'coif5', 'coif8', 'coif10', 'coif9']

    J_settings = [7, 8, 9, 10, 11]  # or any other range of settings
    all_results = []

    # Loop through preprocess functions and settings
    for preprocess, J, wavelet in itertools.product(preprocess_functions, J_settings, wavelets):
        preprocess_name = preprocess.__name__  # Get the preprocess function name
        

        t2c = preprocess(dfC, J, wavelet)
        t2n = preprocess(dfN, J, wavelet)
            
        # Split data
        X = np.vstack([t2c, t2n])
        y = np.hstack([np.ones(len(t2c)), np.zeros(len(t2n))])

        # train/test split                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
            
        
        models = [
            ('LogisticRegression', LogisticRegression()),
            ('Support Vector', SVC()),
            ('Random Forest', RandomForestClassifier()),
            ('XGBoost', XGBClassifier())]
        
        # Loop through models
        for name, model in models:
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            y_test_preds = model.predict(X_test)
            
            # Append new results to the results dataframe
            results = make_results(name, model).round(2)
            results['Preprocess'] = preprocess_name  # Add preprocess name to results
            results['J Setting'] = J  # Add J setting to results
            results['wavelet'] = wavelet
            results['Oversample'] = 0
            all_results.append(results)
            
        # Oversample training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            # Loop through models
        for name, model in models:
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            y_test_preds = model.predict(X_test)
            
            # Append new results to the results dataframe
            results = make_results(name, model).round(2)
            results['Preprocess'] = preprocess_name  # Add preprocess name to results
            results['J Setting'] = J  # Add J setting to results
            results['wavelet'] = wavelet
            results['Oversample'] = 1
            all_results.append(results)
            

    preprocess_functions = [preprocess3, preprocess4]

    # Loop through preprocess functions and settings
    for preprocess, J in itertools.product(preprocess_functions, J_settings):
        preprocess_name = preprocess.__name__  # Get the preprocess function name
        

        t2c = preprocess(dfC, J)
        t2n = preprocess(dfN, J)
            
        # Split data
        X = np.vstack([t2c, t2n])
        y = np.hstack([np.ones(len(t2c)), np.zeros(len(t2n))])

        # train/test split                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
            
        
        models = [
            ('LogisticRegression', LogisticRegression()),
            ('Support Vector', SVC()),
            ('Random Forest', RandomForestClassifier()),
            ('XGBoost', XGBClassifier())]
        
        # Loop through models
        for name, model in models:
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            y_test_preds = model.predict(X_test)
            
            # Append new results to the results dataframe
            results = make_results(name, model).round(2)
            results['Preprocess'] = preprocess_name  # Add preprocess name to results
            results['J Setting'] = J  # Add J setting to results
            results['wavelet'] = 'none'
            results['Oversample'] = 0
            all_results.append(results)
            
        # Oversample training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            # Loop through models
        for name, model in models:
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            y_test_preds = model.predict(X_test)
            
            # Append new results to the results dataframe
            results = make_results(name, model).round(2)
            results['Preprocess'] = preprocess_name  # Add preprocess name to results
            results['J Setting'] = J  # Add J setting to results
            results['wavelet'] = 'none'
            results['Oversample'] = 1
            all_results.append(results)
         
    final_results = pd.concat(all_results)
    final_results.to_csv(f"results/result_{files}.csv",index=False)
    
    print(files)
    files += 1
