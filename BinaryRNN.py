from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

LR = 0.001
LAYERS = 50
EPOCHS = 100
BATCH_SIZE = 5


def exclude_inf(data_frame):
    return data_frame[~data_frame.isin([np.nan, np.inf, -np.inf]).any(1)]


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(data_frame, name):
    le = preprocessing.LabelEncoder()
    data_frame[name] = le.fit_transform(data_frame[name])
    return le.classes_


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(data_frame, target):
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


# Encode a numeric column as zscores
def encode_numeric_zscore(data_frame, name, mean=None, sd=None):
    if mean is None:
        mean = data_frame[name].mean()

    if sd is None:
        sd = data_frame[name].std()

    data_frame[name] = (data_frame[name] - mean) / sd


def standardizing(data_frame, label_name):
    class_ = None
    for each in data_frame.keys():
        if each == label_name:
            class_ = encode_text_index(data_frame, label_name)
        else:
            encode_numeric_zscore(data_frame, each)
    return data_frame, len(class_)


if __name__ == '__main__':
    file_path = './Dataset/2class/data1.csv'
    data = pd.read_csv(file_path)
    data = exclude_inf(data)

    label = 'marker'
    data, class_number = standardizing(data, label)
    # display 5 rows
    data.dropna(inplace=True, axis=1)

    # Break into X (predictors) & y (prediction)
    x, y = to_xy(data, 'marker')
    # x_train.shape=(4156, 1, 117)
    # y_train.shape=(4156, 2)

    '''
    y = data[label].values.reshape(data.shape[0], 1)
    data = data.drop(label, 1)
    data = data.values
    x = data[:, 0:data.shape[1]] 
    
    # x_train.shape=(4156, 1, 117)
    # y_train.shape=(4156, 1)
    '''

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    print("x_train.shape={}".format(x_train.shape))
    print("y_train.shape={}".format(y_train.shape))

    model = Sequential()

    model.add(SimpleRNN(units=50, input_shape=(1, x_train.shape[2])))

    model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
    # 編譯: 選擇損失函數、優化方法及成效衡量方式

    adam = Adam(LR)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=[monitor], verbose=1,
              epochs=EPOCHS)

    # print(model.get_config())

    # Measure accuracy
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(y_test, axis=1)
    auc_score = metrics.accuracy_score(y_eval, pred)
    print("Accuracy: {}".format(auc_score * 100))

    reca_score = metrics.recall_score(y_eval, pred)  # 召回率=真實值為True的情況下預測值仍為True所佔的比例
    print("Recall: {}".format(reca_score * 100))

    prec_score = metrics.precision_score(y_eval, pred)  # 精確率=預測值為True情況下真實值仍為True所佔的比例
    print("Precision: {}".format(prec_score * 100))

    f1_score = metrics.f1_score(y_eval, pred)  # 評估模型穩健程度,越大越好
    print("F-Measure: {}".format(f1_score))
