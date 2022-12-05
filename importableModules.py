#####################################################
### Necessary Imports for all working

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import math
import warnings

### Imports for preprocessing of Data

## Scalers for Scaling Data, not used much by Tree Algorithms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
##LabelEncoder for changing "ÿ" values
from sklearn.preprocessing import LabelEncoder
##Pipelining for easing the WorkFlow
from sklearn.pipeline import PipeLine
## Imputer to fill in values as need be  methods are mean, median
from sklearn.impute import SimpleImputer
## PCA for vectorising all inputs and distributing importances
from sklearn.decomposition import PCA

### Imports for Splits, GridSearch

## Train-Test Split for X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
## KFold for Regression, StratifiedKFold for Classifiction
from sklearn.model_selection import KFold, StratifiedKFold
## GCV, CVS loop internally to reduce coding, scoring in metrics:: ""roc_auc"--classifier,"r2"--regressor
from sklearn.model_selection import GridSearchCV, cross_val_score

### Imports for Classifiers

## KNN:: requires n_neighbors, may have low accuracy at times
from sklearn.neighbors import KNeighborsClassifier
## GNB uses probabilities, not completely accurate-- only for Numerical
from sklearn.naive_bayes import GaussianNB
## GNB uses probabilities, not completely accurate-- only for Categorical
from sklearn.naive_bayes import BernoulliNB
## LogisticRegression uses Log properties for Classifying
from sklearn.linear_model import LogisticRegression
## SGDC uses Learning Rate and a random starting point 'alpha'
from sklearn.linear_model import SGDClassifier
## DA requires all Numeric Data for Classifying
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
## Support Vector has high accuracy and high speed of execution on occasion
from sklearn.svm import SVC
## Tree Based Algorithms are prone to OverFitting, not perfect, but also feature high accuracy
from sklearn.tree import DecisionTreeClassifier

### Imports for Regressors

## KNN:: requires n_neighbors, may have low accuracy at times
from sklearn.neighbors import KNeighborsRegressor
## LinearRegression is fast but requires scaling and can be a bit inaccurate
from sklearn.linear_model import LinearRegression
## Further Types of Linear Regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
## SGDR uses Learning Rate and a random starting point 'alpha'
from sklearn.linear_model import SGDRegressor
## Tree Based Algorithms are prone to OverFitting, not perfect, but also feature high accuracy
from sklearn.tree import DecisionTreeRegressor
