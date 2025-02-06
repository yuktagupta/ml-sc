import pickle 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SelectKBest,chi2 
from sklearn.tree import DecisionTreeClassifier 
df = pd.read_csv('C:\\Users\\Admin\\OneDrive\\Desktop\\pract\\train.csv')
df = pd.read_csv('C:/Users/Admin/OneDrive/Desktop/pract/train.csv')

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True) 
# Step 1 -> train/test/split 
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2,                 
random_state=42) 
X_train.head() 
y_train.sample(5) 
# imputation transformer 
trf1 = ColumnTransformer([ 
    ('impute_age',SimpleImputer(),[2]), 
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6]) 
],remainder='passthrough') 
# one hot encoding 
trf2 = ColumnTransformer([ 
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6]) 
],remainder='passthrough') 
# Scaling 
trf3 = ColumnTransformer([ 
    ('scale',MinMaxScaler(),slice(0,10)) 
]) 
# Feature selection 
trf4 = SelectKBest(score_func=chi2,k=8) 
# train the model 
trf5 = DecisionTreeClassifier() 
pipe = Pipeline([ 
    ('trf1',trf1), 
    ('trf2',trf2), 
    ('trf3',trf3), 
    ('trf4',trf4), 
    ('trf5',trf5) 
]) 
# train 
pipe.fit(X_train,y_train) 
pipe.named_steps 
# Display Pipeline 
from sklearn import set_config 
set_config(display='diagram') 
# Predict 
y_pred = pipe.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy_score(y_test,y_pred) 
# cross validation using cross_val_score 
from sklearn.model_selection import cross_val_score 
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean() 
from sklearn.model_selection import GridSearchCV 
# Corrected parameter grid 
params = { 
 'trf5__max_depth': [1, 2, 3, 4, 5, None] 
} 
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy') 
grid.fit(X_train, y_train) 
grid.best_score_ 
grid.best_params_ 
# export  
pickle.dump(pipe, open('C:/Users/Admin/Desktop/pipe.pkl', 'wb'))

 
#predict.py 
import pickle 
import numpy as np 
import pandas as pd   
#pipe = pickle.load(open('C:/Users/Desktops/pipe.pkl','rb')) 
pipe = pickle.load(open('C:/Users/Admin/Desktop/pipe.pkl', 'rb'))
# Predictions on test data
# Assume user input 
test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7) 
#  Adding a new row to the dataframe 
# test_input2 = np.vstack([ 
#     test_input2,  
#     np.array([12, 'female', 47.0, 0, 0, 54.3, 'C'], dtype=object).reshape(1, 7), 
#     np.array([3, 'male', 23.0, 0, 0, 12.3, 'S'], dtype=object).reshape(1, 7) 
# ]) 
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] 
test_input2_df = pd.DataFrame(test_input2, columns=columns) 
# Assume user input 
print(pipe.predict(test_input2_df)) 