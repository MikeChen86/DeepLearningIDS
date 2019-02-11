import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from DataPreProcessing import exclude_inf, standardizing, load_data

# Hyper Parameters
LR = 0.001
LAYERS = 50
CELL_SIZE = 50
EPOCHS = 100
BATCH_SIZE = 5
OPTIMIZER = Adam(LR)
LOSS_FUNCTION = 'categorical_crossentropy'

if __name__ == '__main__':
    file_path = '../Dataset/multiclass/data2.csv'
    data = pd.read_csv(file_path)

    # Exclude inf Value From DataFrame
    data = exclude_inf(data)

    # Standardizing
    label = 'marker'
    data, class_number = standardizing(data, label)

    # Drop NaN Value
    data.dropna(inplace=True, axis=1)

    # Transform Data Frame to Training Data Format
    (x_train, y_train), (x_test, y_test) = load_data(data, label)

    # Building Model
    model = Sequential()

    # Using LSTM
    model.add(GRU(units=50, input_shape=(1, x_train.shape[2])))

    # Add Output Layer, use Softmax
    model.add(Dense(units=class_number, kernel_initializer='normal', activation='softmax'))

    # Loss function: Cross Entropy
    # Optimizer: Adam
    # Metrics: Accuracy
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])

    # Setting Callback Function: EarlyStopping
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    # Training
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=[monitor], verbose=1,
              epochs=EPOCHS)

    # Measure accuracy
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(y_eval, pred)

    precision, recall, f_score, _ = metrics.precision_recall_fscore_support(y_eval, pred, average='macro')

    print("\n\nMulti Classification GRU:")

    print("Accuracy: {0:.2f}%".format(accuracy * 100))
    print("Precision: {0:.2f}%".format(precision * 100))
    print("Recall: {0:.2f}%".format(recall * 100))
    print("F-Measure: {}".format(f_score))
