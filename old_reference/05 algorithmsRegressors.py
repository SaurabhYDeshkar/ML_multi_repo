###Regressors###

## K-NN Regressor
knn = KneighborsRegressor(n_neighbors='int')

## Linear Regression
lr = LinearRegression()
lr = Ridge(alpha=np.linspace(0.001, 1000))
lr = Lasso(alpha=np.linspace(0.001, 1000))
lr = ElasticNet(alpha=np.linspace(0.001, 1000),  l1_ratio=np.linspace(0, 1))

##SGD
sgd = SGDRegressor(loss='log_loss',random_state=108, eta0=np.linspace(0.001,0.9, 18), 
                    learning_rate=['constant','optimal','invscaling','adaptive'])

##Decision Tree
dtree = DecisionTreeRegressor(random_state=108, max_depth=[None,10,5,3], min_samples_split=[2, 10, 50, 100],
          min_samples_leaf=[1, 10, 50, 100])
##Random Forest
ranfr= RandomForestRegressor(max_features=[2,3,4,5,6], random_state=108, max_depth=[None,10,5,3], min_samples_split=[2, 10, 50, 100],
          min_samples_leaf=[1, 10, 50, 100])

##ENSEMBLING-- VOTING
models = [('Name1', model1), ('Name2', model2), ('Name3', model3), ('Name4',model4)]
voting = VotingRegressor(estimator=models)
##No voting in Regression

##ENSEMBLING-- BAGGING
bagging = BaggingRegressor(base_estimator=model_defined, random_state=108, n_estimators=[10, 50, 100])

##ENSEMBLING-- XG BOOSTING. Better than vanilla Boosting
gbm = xgb.XGBRegressor(random_state=2022, learning_rate=np.linspace(0.001, 1, 10), 
                       max_depth=[2,3,4,5,6], n_estimators=[50,100,150])

##ENSEMBLING--STACKING
stack = StackingRegressor(estimators=models, final_estimator=model4, passthrough=True)
