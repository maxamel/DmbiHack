import pickle
import pandas as pd
import numpy
import scipy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score


def trainGradiantBoostClassifier(dataFrame, Lables,path) :
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=60,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=200,
                 min_samples_leaf=120, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto')
    scores = cross_val_score(clf, dataFrame, Lables)
    print(scores.mean())
    classifier = clf.fit(dataFrame, Lables)
    with open(path + '/xgboost.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    return classifier

def predict(classifier, dataFrame):
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

def loadModelFromFile(path):
    model = None
    with open(path, 'rb') as fid:
        model = pickle.load(fid)
    return model


if __name__ == "__main__":
   filename = r"C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\motion_sensor.csv"
   dataFrame, labData = readDataFile(filename)
   model = trainGradiantBoostClassifier(dataFrame, labData,r'C:\Users\isimkin\Desktop\hakaton\project3\modelsPerIotType\motion_sensor')

   #model= loadModelFromFile(r'C:\Users\isimkin\Desktop\hakaton\project3\modelsPerIotType\lights\xgboost.pkl')
   #df = pd.read_csv(r'C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\lights.csv', nrows=1)
   #columns = df.columns.tolist()
   #cols_to_use = columns[:len(columns) - 1]
   #df_vld = pd.read_csv(r'C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\lights.csv', usecols=cols_to_use, low_memory=False, na_values='?')
   #prediction = predict(model, df_vld)
   #print(prediction)
