
kfold = KFold(n_splits=5, shuffle=True, random_state=108), StratifiedKFold(n_splits=5, shuffle=True, random_state=108)
##
###"models from 04 or 05"###
##
params = {from model.getparams() in Terminal after importing}

gcv = GridSearchCV(Estimator//Ensemble, param_grid=params, scoring='roc_auc/r2/neg_log_loss',
                          cv = kfold, verbose=1/2/3)
gcv.fit(X,y)

print("The Best Score for GCV is:",gcv.best_score_)
print("The Best Parameters are:",gcv.best_params_)

best_model = gcv.best_estimator_
###########
print(best_model.feature_importances_)
ind = np.arange(X.shape[1])
imp = best_model.feature_importances_

i_sorted = np.argsort(-imp)
col_sorted = X.columns[i_sorted]
imp_sorted = imp[i_sorted]

ind = np.arange(X.shape[1])
plt.bar(ind,imp_sorted)
plt.xticks(ind,(col_sorted),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
###########

pd_cv = pd.DataFrame(gcv.cv_results_)
