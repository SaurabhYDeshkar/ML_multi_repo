#### CLASSIFICATION
y_pred_proba=best_model.predict_proba(test_df)

submit=pd.read_csv('path/sample_submission.csv')

submit['TARGET']=y_pred_proba

submit.to_csv('submit_sgd.csv', index=False)

#### REGRESSION
pd_y_pred=pd.DataFrame(y_pred_proba,columns=list(lab_enc.classes_))

submit=pd.read_csv('path/sampleSubmission.csv')

submit_lr=pd.concat([submit['id'],pd_y_pred],axis=1)

submit_lr.to_csv('submit_sgd.csv', index=False)
