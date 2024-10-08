###########################
###VOTING####
from sklearn.ensemble import VotingClassifier, VotingRegressor
###BAGGING####
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
###BOOSTING####
##GBM--- Slow
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
##XGB--- Faster operation, requires different library
from xgboost import XGBClassifier, XGBRegressor
###STACKING####
from sklearn.ensemble import StackingClassifier, StackingRegressor
####ISOLATION FOREST####
from sklearn.ensemble import IsolationForest
