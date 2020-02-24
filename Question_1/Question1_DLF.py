#!/usr/bin/env python
# coding: utf-8

# <div style="width: 400px; height: 160px;">
#     <img src="rplogo_small.png" width="100%" height="100%" align="left">
# </div>
# 
# ###     TIPP - AAI Assignement (Deep Learning Fundamentals)<br>Due Date: 26 February 2020
# ###     Submitted By: <u>KOAY</u> SENG TIAN<br>Email: sengtian@yahoo.com
# 

# In[1]:


# TIPP - AAI Assignment (Deep Learning Fundamentals)
# Date Due: 26 February 2020
# Submited By: KOAY SENG TIAN
# Email: sengtian@yahoo.com
#
# GitHub: https://github.com/koayst/rp_deeplearning_assignment
#
# Note: source of below statement => sonar.names
# https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
# Gorman and Sejnowski further report that a nearest neighbor classifier on
# the same data gave an 82.7% probability of correct classification.

from keras import models
from keras import layers
from keras import losses
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import os
import pandas as pd

# set max lines Pandas can display
pd.set_option('display.max_rows', 220)

# for reproducibility, 1337 is arbitrarily set
np.random.seed(1337)

# verbose mode - 0=show little info, 1=show more info like charts
#verbose=0
verbose=1


# In[2]:


def load_data():
    # the dataset files are stored under the 'data' directory
    filedir = os.path.join(os.getcwd(), 'data')
    mines_filename = 'sonar.mines'
    rocks_filename = 'sonar.rocks'
    sonar_filename = 'sonar.all-data'

    # load the data file
    file = os.path.join(filedir, sonar_filename)
    df = pd.read_csv(file, sep=',', header=None)
    
    # create the header for the dataframe
    # the header starts with 'A' followed by a number
    header = [f"A{x:02d}" for x in range(1, df.shape[1])]
    # the last column is named as 'C' as in 'Class'
    header.append('C')
    df.columns = header
    
    return df


# In[3]:


def exploratory_data_analysis(df):
    print('Any null ?', end=' ')
    print(df.isnull().values.any())
    print()
    print('NULL count in each column:')
    print(df.isnull().sum())
    print()
    print('Any NaN ?', end=' ')
    print(sonar_df.isna().any().any())
    print()
    print('ZERO count in each column:')
    # columns 42 to 59 have 0 values, but it is still OK
    # as the document said 'each pattern is a set of 60 
    # numbers in the range 0.0 to 1.0 [file: sonar.names]
    print(df.eq(0).sum())
    print()
    
    # True if the dtype is object (categorical), otherwise False
    mask = sonar_df.dtypes == np.object
    # Extract column names that are categorical
    categorical_cols = sonar_df.columns[mask]
    print('What are the categorical column(s)?', end=' ')
    print(categorical_cols)
    print()
   
    # Extract categorical data
    categorical_data = sonar_df.select_dtypes(include=['object']).copy()
    
    # Count the number of cateory for each column
    print('Unique value count of categorical columns:')
    unique_values_counts = (categorical_data.apply(lambda x: x.nunique()).sort_values(ascending=False))
    print(unique_values_counts)
    print()
    
    print('Unique value of categorical columns:')
    print(categorical_data.apply(pd.Series.value_counts))


# In[4]:


def charts(df):
    df.plot.box(figsize=(12,7))
    plt.xticks(np.arange(1, 60, 3), [f"A{x:02d}" for x in range(1, 60, 3)], rotation=45)
    plt.title('Fig 1: Boxplot for A01 - A60 Angles')
    plt.show()


# In[5]:


# load the dataset and observe its shape
sonar_df = load_data()
sonar_df.shape

if verbose==1:
    print(sonar_df.head())
    print(sonar_df.tail())
    print()
    print(sonar_df.info())
    print()
    
    # print to observe the min, max values of dataset
    descT = sonar_df.describe().transpose()
    cols = list(descT)

    #move 'max' column next to 'min' column for easy visual comparison
    cols.insert(cols.index('25%'), cols.pop(cols.index('max')))
    descT = descT.loc[:, cols]
    print(descT)
    print()

    charts(sonar_df)


# In[6]:


# looking at Fig 1. above, the data is normally distributed

if verbose==1:
    exploratory_data_analysis(sonar_df)


# In[7]:


X = sonar_df.drop(columns=['C'])
y = sonar_df['C']

if verbose==1:
    X.head()
    y.head()


# In[8]:


# perform label encoding on last column 
lbl_encoder = LabelEncoder()

# encode 'R' as in 'Rock' to 0
# encode 'M' as in 'Mine' to 1
lbl_encoder = LabelEncoder.fit(lbl_encoder, y = ["R", "M"])
y_encoded = lbl_encoder.fit_transform(y)

# perform one hot encoding
y = to_categorical(y_encoded)

# cast X to numpy array
X = X.to_numpy()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)


# In[10]:


# perform scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform (X_test)


# In[11]:


# check to ensure the shape and 
# data type are what I think it should be
if verbose==1:
    X_train.shape
    type(X_train)
    X_test.shape
    type(X_test)
    y_train.shape
    type(y_train)
    y_test.shape
    type(y_test)


# In[12]:


from keras import regularizers

# build the Artifical Neuro Network layers
network = models.Sequential()
network.add(layers.Dense(28, activation='relu', input_shape=(60,)))
network.add(layers.Dense(28, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
network.add(layers.Dense(2, activation='softmax'))


# In[13]:


network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.summary()


# In[14]:


history = network.fit(
                      X_train, y_train, validation_split=0.1, 
                      epochs=70, batch_size=6, shuffle=True, verbose=verbose
                     )

print()
test_loss, test_acc = network.evaluate(X_test, y_test)

print()
print('test_acc:', test_acc)


# In[15]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)


# In[16]:


fig = plt.figure(figsize=(11,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#draw chart to show validation vs training losses
ax1.plot(epochs, loss_values, 'r', label='Training loss')
ax1.plot(epochs, val_loss_values, 'b', label='Validation loss')
ax1.title.set_text('Fig 2. Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# draw chart to show validation vs training accuracy
ax2.plot(epochs, acc_values, 'r', label='Training acc')
ax2.plot(epochs, val_acc_values, 'b', label='Validation acc')
ax2.title.set_text('Fig 3. Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.show()


# In[ ]:




