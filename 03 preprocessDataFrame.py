var_df= pd.read_csv(path="", index_col=0) ##The first Column can be Index or irrelevant, change it to INDEX_COL
##Setting Dummies if OUTPUT Data has incompatible values, convert to Binary when Classifying
dum_vardf= pd.get_dummies(var_df, drop_first=True)
## Dropping unnecessary Columns from the X Dataset
X = dum_vardf.drop('output',axis=1) ##Dropping Column for Prediction Value
y = dum_vardf['output'] ##Making the Predictions column as a Column Matrix

##If X is needed to be reshaped from a Row Vector to Column Vector
X = X.values
X = X.reshape(-1,1)
##Using Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 108, 
                                                    test_size= 0.27, 
                                                    shuffle= True, stratify= y)
