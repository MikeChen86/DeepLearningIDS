from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def exclude_inf(data_frame):
    """
    Exclude inf value from dataset
    :param data_frame: Dataset
    :return: Dataset without inf value
    """
    return data_frame[~data_frame.isin([np.nan, np.inf, -np.inf]).any(1)]


def encode_text_index(data_frame, name):
    """
    Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
    :param data_frame: Dataset
    :param name: Header name of label
    :return: Number of classes
    """
    le = preprocessing.LabelEncoder()
    data_frame[name] = le.fit_transform(data_frame[name])
    return le.classes_


def to_xy(data_frame, target):
    """
    Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
    :param data_frame: Dataset
    :param target: Header name of label
    :return: Training data and labels
    """
    result = []
    for x in data_frame.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = data_frame[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(data_frame[target])
        return data_frame.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return data_frame.as_matrix(result).astype(np.float32), data_frame.as_matrix([target]).astype(np.float32)


def encode_numeric_zscore(data_frame, name, mean=None, sd=None):
    """
    Encode a numeric column as z-scores
    :param data_frame: Dataset
    :param name: Header name of column
    :param mean: Mean of column data
    :param sd: Standard Deviation of column data
    """
    if mean is None:
        mean = data_frame[name].mean()

    if sd is None:
        sd = data_frame[name].std()

    data_frame[name] = (data_frame[name] - mean) / sd


def standardizing(data_frame, label_name):
    """
    Standardizing
    :param data_frame: Dataset
    :param label_name: Header name of label
    :return: Dataset after Standardizing, number of classes
    """
    class_ = None
    for each in data_frame.keys():
        if each == label_name:
            class_ = encode_text_index(data_frame, label_name)
        else:
            encode_numeric_zscore(data_frame, each)
    return data_frame, len(class_)


def load_data(data_frame, label_name, test_size=0.2, random_state=42):
    x, y = to_xy(data_frame, label_name)
    '''
    y = data[label].values.reshape(data.shape[0], 1)
    data = data.drop(label, 1)
    data = data.values
    x = data[:, 0:data.shape[1]] 

    # x_train.shape=(4156, 1, 117)
    # y_train.shape=(4156, 1)
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    pass
