var_df= pd.read_csv(path="", index_col=0) ##The first Column can be Index or irrelevant, change it to INDEX_COL
##Setting Dummies if OUTPUT Data has incompatible values, convert to Binary when Classifying
dum_vardf= pd.get_dummies(var_df, drop_first=True)
## Dropping unnecessary Columns from the X Dataset
X = dum_vardf.drop('output',axis=1) ##Dropping Column for Prediction Value
y = dum_vardf['output'] ##Making the Predictions column as a Column Matrix
