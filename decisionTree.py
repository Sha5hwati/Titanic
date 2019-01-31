from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

import sys
import os.path
import pandas as pd
import numpy as np

dataset = './dataset/titanic.csv'
use = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]

def get_titanic_data():
    if os.path.isfile(dataset):
        print("Loading titanic dataset ...")
        data = pd.read_csv(dataset, usecols = use)
        print("\nDataset loaded successfully\n\n")
        return data
    else:
        print('File not found')
        print('\n\nExiting...')
        sys.exit()
        

print("1- Getting data")
titanic = get_titanic_data()
print(titanic.head())
print("-------------------------\n\n")

print("2- Preprocessing")
clean = preprocessing.LabelEncoder()
titanic["sex"] = clean.fit_transform(titanic["sex"])
titanic["embarked"] = clean.fit_transform(titanic["embarked"].astype(str))
titanic[use] = titanic[use].replace(np.NaN, 0)

print("\n\n")

print("3- Making features and target attributes")
features = titanic[["pclass", "sex", "age", "sibsp", "parch", "embarked"]]
target = titanic["survived"]
print("\n\n")

print("4- Making the train and test set (80-20)")
feat_train, feat_test, tar_train, tar_test = train_test_split(features, target, test_size=0.20, random_state=2)

print("\nTraining features - ")
print(feat_train.head())
print("-------------------------\n\n")

print("5- Building the decision tree")
clf = DecisionTreeClassifier(max_depth = 6, random_state=1)

clf = clf.fit(feat_train, tar_train)

y_pred = clf.predict(feat_test)

accuracy = accuracy_score(y_pred, tar_test)

print("Accuracy: ", accuracy)
print("-------------------------\n\n")

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, 
                feature_names = ["pclass", "sex", "age", "sibsp", "parch", "embarked"], 
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic_tree.png')
Image(graph.create_png())
