import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def crossSetCreation(dataframe,lables):
    X_train, X_test, y_train, y_test = train_test_split( dataframe, lables, test_size = 0.4, random_state = 0)
    return  X_train, X_test, y_train, y_test

def calcModelWeigths( X_test, y_test,models):
    predVector = []
    for model in models :
        predVector.append( model.predict(X_test))

    print("ttt")


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
    with open('path', 'rb') as fid:
        model = pickle.load(fid)
    return model

if __name__ == "__main__":
    filepath = r"C:\Users\isimkin\Desktop\hakaton\project3\byDevice\lights.csv"
    featuresData, labData = readDataFile(filepath)

    models = [loadModelFromFile]
    X_train, X_test, y_train, y_test =  crossSetCreation(featuresData,labData)
    calcModelWeigths(X_test, y_test)
    print("done")
