import pandas as pd
import numpy
import scipy
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.feature_selection import chi2

def readDataFile(path):
    df = pd.read_csv(path, nrows=1)  # read just first line for columns
    columns = df.columns.tolist()  # get the columns
    cols_to_use = columns[:len(columns) - 1]  # drop the last one
    lablesCols = columns[len(columns) - 1:]
    featuresData = pd.read_csv(path, usecols=cols_to_use,delimiter=',',header=0)
    labData = pd.read_csv(path, usecols=lablesCols, delimiter=',', header=0)
    return featuresData,labData


def selectBestFeatures(dataset,lables,numberOfFeatures):
    # df21 = dataset.head(500)
    # df22 = dataset.tail(500)
    # df2 = pd.concat([df21,df22])
    # df31 = lables.head(500)
    # df32 = lables.tail(500)
    # df3 = pd.concat([df31, df32])
    select_k_best_classifier = SelectKBest(score_func=chi2, k=numberOfFeatures).fit(dataset, lables)
    #select_k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=numberOfFeatures).fit(dataset, lables)
    mask = select_k_best_classifier.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, dataset.columns.values.tolist()):
        if bool:
            new_features.append(feature)
    dataframe = dataset[new_features]
    return dataframe



def remove_apprentice(sorceFile,destFile) :
    with open(sorceFile, 'r') as infile:
        with open(destFile, 'w') as outfile:
            data = infile.read()

            data = data.replace('\"', "")
            data = data.replace('?', "0.5")
            outfile.write(data)


if __name__ == "__main__":
    filename = "C:/Users/isimkin/Desktop/hakaton/project3/byDevice/security_camera.csv"
    destFile = "C:/Users/isimkin/Desktop/hakaton/project3/byDevice/security_camera111.csv"
    numberOfFeatures = 50
    #remove_apprentice(filename,destFile)
    dataSet,lables = readDataFile(destFile)
    print(dataSet.shape)
    print(lables.shape)

    result = selectBestFeatures(dataSet,lables,numberOfFeatures)
    lables.columns = ["device_category"]
    result = pd.concat([result,lables],axis=1)
    # pd.to_csv(path= "C:/Users/isimkin/Desktop/hakaton/project3/dataSets/hackathon_IoT_training_set_based_on_01mar2017WithoutApp222.csv")
    result.to_csv("C:/Users/isimkin/Desktop/hakaton/project3/withSelectedFeatures/security_camera.csv", sep=',', encoding='utf-8', index=False)
    print("done")
    # get_top_n_features(dataSet, numberOfFeatures)
    # generator = generator(filename=filename)

    #
