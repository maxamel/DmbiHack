# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
#
# X, y = make_classification(n_samples=1000, n_features=4,
#                           n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(X, y)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
# print(clf.feature_importances_)
# [ 0.17287856  0.80608704  0.01884792  0.00218648]
# >>> print(clf.predict([[0, 0, 0, 0]]))
import pandas as pd
import numpy
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score


def trainRF(dataFrame,Lables) :
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_val_score(clf, dataFrame, Lables)
    print(scores.mean())

    classifier = clf.fit(dataFrame, Lables)
    return classifier

def predictWithRF(classifier,dataFrame):
    predictions = classifier.predict_proba(dataFrame)
    return predictions



def readDataFile(path):
    df = pd.read_csv(path, nrows=1)  # read just first line for columns
    columns = df.columns.tolist()  # get the columns
    cols_to_use = columns[:len(columns) - 1]  # drop the last one
    lablesCols = columns[len(columns) - 1:]
    featuresData = pd.read_csv(path, usecols=cols_to_use,delimiter=',',header=0)
    labData = pd.read_csv(path, usecols=lablesCols, delimiter=',', header=0)
    return featuresData,labData


if __name__ == "__main__":
    filename = "C:/Users/isimkin/Desktop/hakaton/project3/withSelectesdFeatures/security_camera.csv"
    dataFrame = readDataFile(filename)
