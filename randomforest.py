
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def trainRF(dataFrame, Lables, path):
    clf = RandomForestClassifier(n_estimators=200,
                                 criterion="gini",
                                 max_depth=None,
                                 min_samples_split=220,
                                 min_samples_leaf=200,
                                 min_weight_fraction_leaf=0.,
                                 max_features="auto",
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.,
                                 min_impurity_split=None,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=6,
                                 random_state=None,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None)
    scores = cross_val_score(clf, dataFrame, Lables)
    print(scores.mean())
    clf.fit(dataFrame, Lables)
    with open(path+'/rf.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print("classifierReady: ")
    return clf


def predict(classifier, dataFrame):
    predictions = classifier.predict_proba(dataFrame)
    return predictions


def readDataFile(path):
    df = pd.read_csv(path, nrows=1)  # read just first line for columns
    columns = df.columns.tolist()  # get the columns
    cols_to_use = columns[:len(columns) - 1]  # drop the last one
    lablesCols = columns[len(columns) - 1:]
    featuresData = pd.read_csv(path, dtype=float, usecols=cols_to_use, delimiter=',', header=0)
    labData = pd.read_csv(path, usecols=lablesCols, delimiter=',', header=0)
    return featuresData, labData


if __name__ == "__main__":
    filename = r"C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\motion_sensor.csv"
    dataFrame, labData = readDataFile(filename)
    model = trainRF(dataFrame, labData, r'C:\Users\isimkin\Desktop\hakaton\project3\modelsPerIotType\motion_sensor')
    # df = pd.read_csv(r'C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\lights.csv', nrows=1)
    # columns = df.columns.tolist()
    # cols_to_use = columns[:len(columns) - 1]
    # df_vld = pd.read_csv(r'C:\Users\isimkin\Desktop\hakaton\project3\withSelectedFeatures\lights.csv',
    #                      usecols=cols_to_use, low_memory=False, na_values='?')
    #
    # prediction = predict(model, df_vld)
    # print(prediction)
