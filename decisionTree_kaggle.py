from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

import sys, csv
import os.path
import pandas as pd
import numpy as np

train = './titanic/train.csv'
test = './titanic/test.csv'
train_use = ["Pclass", "Sex", "Age", "SibSp", "Embarked", "Survived"] #"Parch", 
test_use = ["Pclass", "Sex", "Age","Embarked", "SibSp",]

def get_train_titanic_data():
    if os.path.isfile(train):
        print("Loading " + str(train) + " dataset ...")
        data = pd.read_csv(train, usecols = train_use)
        print("\nDataset loaded successfully\n\n")
        return data
    else:
        print('File not found')
        print('\n\nExiting...')
        sys.exit()
        
def get_test_titanic_data():
    if os.path.isfile(test):
        print("Loading " + str(test) + " dataset ...")
        data = pd.read_csv(test, usecols = test_use)
        print("\nDataset loaded successfully\n\n")
        return data
    else:
        print('File not found')
        print('\n\nExiting...')
        sys.exit()
        
        
def train_preprocess(dataset):
    clean = preprocessing.LabelEncoder()
    dataset["Sex"] = clean.fit_transform(dataset["Sex"])
    dataset["Embarked"] = clean.fit_transform(dataset["Embarked"].astype(str))
    dataset[train_use] = dataset[train_use].replace(np.NaN, 0)
    return dataset

def test_preprocess(dataset):
    clean = preprocessing.LabelEncoder()
    dataset["Sex"] = clean.fit_transform(dataset["Sex"])
    dataset["Embarked"] = clean.fit_transform(dataset["Embarked"].astype(str))
    dataset[test_use] = dataset[test_use].replace(np.NaN, 0)
    return dataset

def get_features_target(dataset):
    features = dataset[test_use]
    target = dataset["Survived"]
    return features, target
        

print("1- Getting data")
titanic = get_train_titanic_data()
print(titanic.head())
print("-------------------------\n\n")

print("2- Preprocessing")
titanic = train_preprocess(titanic)
print(titanic.head())
print("\n\n")

print("3- Making features and target attributes")
features, target = get_features_target(titanic)
print("\n\n")

print("4- Building the decision tree")
clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf=0.07, random_state=1)

clf = clf.fit(features, target)
print("\n\n")

print("5- Getting test data")
test_use.append("PassengerId")
test = get_test_titanic_data()
test = test_preprocess(test)
print(test.head())
print("-------------------------\n\n")

print("6- Get test features and target")
passengers = test["PassengerId"]
test_use.remove("PassengerId")
test_features = test[test_use]
print(test_features.head())
print("\n\n")

y_pred = clf.predict(test_features)

p = clf.predict(titanic[test_use])
##Overfitting
print("Accuracy: ", accuracy_score(target, p))

open_file = open("./result.csv", 'w')
writer = csv.writer(open_file,  delimiter=',', lineterminator='\n')
title = ['PassengerId', 'Survived']
writer.writerow(title)
for i in range(len(y_pred)):
    row = [passengers[i], y_pred[i]]
    writer.writerow(row)
open_file.close()
print("-------------------------\n\n")

#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
#                special_characters=True, 
#                feature_names = ["pclass", "sex", "age", "sibsp", "parch", "embarked"], 
#                class_names=['0', '1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('titanic_tree.png')
#Image(graph.create_png())
