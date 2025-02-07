import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import lightgbm as lgbm

df = pd.read_csv(r"C:\Users\shara\Downloads\heart disease dataset\cleaned_merged_heart_dataset.csv")

x = df.drop(columns= 'target', axis = 1)
y = df['target']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.2, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

models = {
    'LogisticRegression': LogisticRegression(max_iter= 1000),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'DecisionTreeClassifier' : DecisionTreeClassifier(),
    'KNeighbourClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(),
    'BernoulliNB': BernoulliNB(),
    
}

results = []

for name, model in models.items():
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    
    acc = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Confusion Matrix': cm
    })
    
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model, file)

results_model = pd.DataFrame(results)
results_model.to_csv('Model_Eval.csv')