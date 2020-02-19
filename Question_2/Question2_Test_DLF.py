#!/usr/bin/env python
# coding: utf-8

# <div style="width: 400px; height: 160px;">
#     <img src="rplogo_small.png" width="100%" height="100%" align="left">
# </div>
# 
# ###     TIPP - AAI Assignement (Deep Learning Fundamentals)<br>Due Date: 21 February 2020
# ###     Submitted By: <u>KOAY</u> SENG TIAN<br>Email: sengtian@yahoo.com
# 

# ## Question 2 (Testing #1 & Testing #2)

# In[ ]:


import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

# set verbose=0 the skip the charts and other information
# set verbose=1 to see chart and other information
#verbose=0
verbose=1

# uncomment one of the 2 lines to test either datatest.txt or datatest2.txt
testdata_file = 'datatest.txt'
#testdata_file = 'datatest2.txt'


# In[ ]:


# You can run this either as a 1) jupyter note or 2) File -> Download as -> Python (.py) 
#
# 1) Jupyter notebook: 
#    Settings: 1) set verbose (see above) to either 0 (no charts) or 1 (charts and other information)
#              2) comment / uncomment test_datafile to use datatest.txt or datatest2.txt to test the model
#
# 2) Python script (.py):
#        1) open a terminal (command prompt in windows that has python interpreter)
#        2) python Question2_Test_DLF.py -h
# 
#           usage: Question #2 Testing [-h] [-v VERBOSE] [-t TEST]
#
#           optional arguments:
#              -h, --help            show this help message and exit
#              -v VERBOSE, --verbose VERBOSE
#                                   turn on or off verbose mode (default: 1)
#              -t TEST, --test TEST  (0) for datatest.txt and (1) datatest1.txt
#        3) Example: python Question2_Test_DLF.py -v 0 -t 1 OR
#                    python Question2_Test_DLF.py -v 1 -t 2
#
if __name__ == '__main__' and '__file__' in globals():
    ap = argparse.ArgumentParser('Question #2 Testing')
    ap.add_argument('-v', '--verbose', default='0', type=int, help='turn off(0, default) or on(1) verbose mode')
    ap.add_argument('-t', '--test', default='0', type=int, help='(0, default) for datatest.txt and (1) datatest1.txt')

    args = vars (ap.parse_args())

    verbose = 0
    if args.get('verbose') == 1:
        verbose=1
        
    if args.get('test') == 1:
        testdata_file = 'datatest2.txt'
        
    print('Verbose=', verbose, ' Test data file=', testdata_file)
    print()
    


# In[ ]:


import keras

model_filename = 'model.pkl'
scaler_filename = 'scaler.pkl'

model_dir = os.path.join(os.getcwd(), 'model')
data_dir = os.path.join(os.getcwd(), 'data')


# In[ ]:


# load model and scaler
with open(os.path.join(model_dir, model_filename), 'rb') as model_file:
    model = pickle.load(model_file)
    
with open(os.path.join(model_dir, scaler_filename), 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# In[ ]:


test_df = pd.read_csv(os.path.join(data_dir, testdata_file))
test_df['date']= pd.to_datetime(test_df['date']) 
test_df.sort_values(by='date', inplace=True, ascending=True)

if verbose==1:
    print('Test data shape=', test_df.shape)
    print()
    print('Training Data:')
    print('The time series starts from: ', test_df.date.min())
    print('The time series ends on: ', test_df.date.max())
    print()


# In[ ]:


import mycharts

if verbose==1:
    mycharts.chart01(test_df.columns[1:], test_df, 'blue')


# In[ ]:


#set the dat as index
test_df = test_df.set_index('date')

# shift the data forward by 1
# to simulate the effect for future occupancy forecasting 
test_df = test_df.shift(1)

# drop the first row
test_df.drop(test_df.head(1).index, inplace=True)

if verbose==1:
    print('First row was dropped')
    print('Shape=', test_df.shape)
    print()
    


# In[ ]:


X = test_df.iloc[:,  : -1].copy().to_numpy()
y = test_df.iloc[:, -1 ].copy().to_numpy()


# In[ ]:


X = scaler.transform(X)

#reshape for LSTM's format samples, timestep and features
X = X.reshape(-1, 1, 5)


# In[ ]:


pred = model.predict(X)
pred_classes = model.predict_classes(X)


# In[ ]:


pred_classes_squeezed = np.squeeze(pred_classes)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print('\n-----------------------------------------------------')
print('Test file: ', testdata_file)
print('-----------------------------------------------------')
print('Confusion Matrix:')
print(pd.crosstab(y, pred_classes_squeezed, rownames=['True'], colnames=['Predicted'], margins=True))
print('\n-----------------------------------------------------')
print('Classification Report:')
print(classification_report(y, pred_classes_squeezed))


# In[ ]:


# using datatest.txt, the accuracy is 98%, precision (0=100%, 1.0=96%)
# using datatest2.txt, the accuracy is 99%, precision (0=100%, 1.0=96%)


# In[ ]:




