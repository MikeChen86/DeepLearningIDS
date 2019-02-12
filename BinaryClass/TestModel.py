import pandas as pd
import numpy as np
from DataPreProcessing import exclude_inf, standardizing, to_xy
from keras.models import load_model

if __name__ == '__main__':
    model = load_model('BinaryRNN.h5')
    file_path = '../Dataset/2class/data1.csv'
    data = pd.read_csv(file_path)

    # exclude inf value from DataFrame
    data = exclude_inf(data)

    # Standardizing
    label = 'marker'
    data, class_number = standardizing(data, label)

    # Drop NaN value
    data.fillna(value=0, inplace=True)

    # Transform Data Frame to Training Data Format
    x, y = to_xy(data, label)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

    score = model.evaluate(x, y, verbose=0)
    print('Accuracy: {}%'.format(score[1] * 100))
