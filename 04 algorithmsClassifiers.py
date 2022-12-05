###Classifiers###

## K-NN Classifier
knn = KneighborsClassifier(n_neighbors='int')

## Naive-Bayes Bernoulli for ONLY Categorical Data, Gaussian works with Numerical Data as well
nb = BernoulliNB(), GaussianNB()

## Logistic Regression-- Classification variation of Linear Regression
logreg = LogisticRegression()

##SGD
sgd = SGDClassifier(loss='log_loss',random_state=108, eta0=np.linspace(0.001,0.9, 18), 
                    learning_rate=['constant','optimal','invscaling','adaptive'])

##Discriminant Analysis
da = LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()

##Support Vector Machine-- Support Vector Classifier, C is L2-penalty
##kernel{‘linear’, ‘poly’, ‘rbf’}
svm = SVC(probability=True, kernel=..., C=np.linspace(0.001,0.9, 18), gamma={‘scale’, ‘auto’} or np.linspace(0.001, 9, 18))

##Decision Tree
dtree = DecisionTreeClassifier(random_state=108, max_depth=[None,10,5,3], min_samples_split=[2, 10, 50, 100],
          min_samples_leaf=[1, 10, 50, 100])
##Random Forest
ranfr= RandomForestClassifier(max_features=[2,3,4,5,6], random_state=108, max_depth=[None,10,5,3], min_samples_split=[2, 10, 50, 100],
          min_samples_leaf=[1, 10, 50, 100])

##ENSEMBLING-- VOTING
models = [('Name1', model1),('Name2', model2),('Name3', model3)]
voting = VotingClassifier(estimator=models, voting={'hard','soft'}, weights=>array<)
##Hard voting is not preferred, Soft voting gives better results

##ENSEMBLING-- BAGGING
bagging = BaggingClassifier(base_estimator=model_defined, random_state=108, n_estimators=[10, 50, 100])

##ENSEMBLING-- XG Boosting. Better than simple Boosting
gbm = xgb.XGBClassifier(random_state=2022, learning_rate=np.linspace(0.001, 1, 10), 
                       max_depth=[2,3,4,5,6], n_estimators=[50,100,150])
