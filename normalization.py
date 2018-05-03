import pandas
import numpy as np
from sklearn import preprocessing


def fix_series_data(series, means_data):
    return series.replace(np.nan, means_data)


def str2float(s):
    try:
        f = np.float(s)
    except ValueError:
        f = np.nan
    except TypeError:
        f = np.nan
    return np.float(f)


def clean_str(df):
    for col in df.iloc[:, :-1]:
        df[col] = df[col].apply(str2float)
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(np.float32)
    return df


def fix_data_frame(data_frame, cols):
    data_frame = clean_str(data_frame)
    means = np.nanmean(data_frame, axis=0)

    fix_data = pandas.DataFrame()
    for col, val in enumerate(cols):
        fix_data = pandas.concat([fix_data, fix_series_data(data_frame[val], means[col])], axis=1)
    return fix_data


def normalize_data(path):
    data = pandas.read_csv(path)
    categories = data[['device_category']]
    data_to_normalize = data.drop(['device_category'], axis=1)
    fix_data_to_normalize = fix_data_frame(data_to_normalize, data.columns[:-1])
    min_max_scalar = preprocessing.MinMaxScaler()
    scaled = pandas.DataFrame(min_max_scalar.fit_transform(fix_data_to_normalize))
    normalize_data = pandas.concat([scaled, categories], axis=1)
    normalize_data.columns = data.columns

    return normalize_data


if __name__ == "__main__":
    returned = normalize_data("trainingset.csv")
    returned.to_csv("trainingset_result.csv", encoding='utf-8', sep=',', index=False)
