import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively

from sklearn import tree
clf_min_2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf_min_50 = tree.DecisionTreeClassifier(min_samples_split=50)

clf_min_2 = clf_min_2.fit(features_train, labels_train)
pred_min_2 = clf_min_2.predict(features_test)

clf_min_50 = clf_min_50.fit(features_train, labels_train)
pred_min_50 = clf_min_50.predict(features_test)


from sklearn.metrics import accuracy_score
acc_min_samples_split_2 = accuracy_score(pred_min_2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred_min_50, labels_test)



def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

#output:
#{"message": "{'acc_min_samples_split_50': 0.912, 'acc_min_samples_split_2': 0.908}"}
