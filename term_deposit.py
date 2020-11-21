
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Importing the data
data = pd.read_csv("term-deposit-marketing-2020.csv")
data.head()

#%%seperating categorical and numerical
data["output"]=(data.y=="yes").astype("int")

numerical_columns=["age","balance","day","duration","campaign"]

categorical_columnns=["job","marital","education","default","housing","loan","contact","month"]

#%% one-hot encoding
encoded_categorical_columns=pd.get_dummies(data[categorical_columnns])

data=pd.concat([data,encoded_categorical_columns],axis=1)

#%%creating a new dataset
all_cat_columns_stored=list(encoded_categorical_columns.columns)

data_1=numerical_columns+all_cat_columns_stored

new_data=data[data_1+["output"]]

#%% data splitting

X_train, X_test, y_train, y_test = train_test_split(new_data, data["output"], test_size=0.33, random_state=42)

X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values



#%%KNN classification

from sklearn.neighbors import KNeighborsClassifier
knn =  KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print(" {} nn score: {} ".format(3,knn.score(X_test,y_test)))


#%% cross-validation
from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=knn,X=X_train,y=y_train,cv=5)
print(acc.mean()*100)

#%% finding the most seperable features using feature selection
from skfeature.function.statistical_based import f_score

correct = 0
num_fea = 10

score = f_score.f_score(X_train, y_train)
idx = f_score.feature_ranking(score)
selected_features = X_train[:, idx[0:num_fea]]
print([selected_features])


      
