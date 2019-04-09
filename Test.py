import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.activations import softmax
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from DataPreProcessing import exclude_inf, standardizing, load_data
import csv
from sklearn.model_selection import StratifiedKFold

LR = 0.001
LAYERS = 5
CELL_SIZE = 50
EPOCHS = 100
BATCH_SIZE = 5
TIME_STEP = 4
OPTIMIZER = Adam(LR)
LOSS_FUNCTION = 'categorical_crossentropy'

if __name__ == '__main__':
    classifiers = [
        ('RandomForest', 'RF', RandomForestClassifier()),
        ('KNN', 'KNN', KNeighborsClassifier()),
        ('SVM', 'SVM', SVC(gamma='auto')),
        ('GaussianNaiveBayes', 'GaussianNB', GaussianNB()),
        ('DecisionTree', 'DT', DecisionTreeClassifier()),
        ('KMeans', 'KMeans', KMeans(n_clusters=2, random_state=0)),
        ('AdaBoost', 'AdaBoost', AdaBoostClassifier()),
        ('LogisticRegression', 'LR', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'))
    ]
    '''
    data = dict()
    split_index = dict()
    class_number = dict()
    for type_ in ['Binary', 'Triple', 'Multi']:
        data[type_] = list()
        split_index[type_] = list()
        for data_set in range(1, 16):
            file_path = 'Dataset/{}class/data{}.csv'.format(type_, data_set)
            data_ = pd.read_csv(file_path)

            data_ = exclude_inf(data_)

            label = 'marker'
            data_, class_number[type_] = standardizing(data_, label)

            data_.dropna(inplace=True, axis=1)
            data_ = load_data(data_, label)
            data[type_].append(data_)
            skf = StratifiedKFold(n_splits=10)
            split_index[type_].append(skf.split(data_[0], data_[1]))
    '''
    for path, abbreviation, model in classifiers[2:3]:

        for type_ in ['Binary', 'Triple', 'Multi']:
            print("Class: {} Model:{}".format(type_, path))
            with open('result/{}/{}{}_total.csv'.format(path, type_, abbreviation), 'w', newline='') as csv_file:
                header = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F-Measure']
                writer = csv.writer(csv_file)
                writer.writerow(header)
                '''
                for each in range(1, 16):
                    file_path = 'Dataset/{}class/data{}.csv'.format(type_, each)
                    data_ = pd.read_csv(file_path)

                    data_ = exclude_inf(data_)

                    label = 'marker'
                    data_, class_number = standardizing(data_, label)

                    data_.dropna(inplace=True, axis=1)
                    data_ = load_data(data_, label)
                    skf = StratifiedKFold(n_splits=10)

                    accuracy_avg = 0
                    precision_avg = 0
                    recall_avg = 0
                    f_measure_avg = 0
                    if path == 'KMeans':
                        model = KMeans(n_clusters=class_number, random_state=0)

                    for train_index, test_index in skf.split(data_[0], data_[1]):
                        x_train = data_[0][train_index]
                        y_train = data_[1][train_index]
                        x_test = data_[0][test_index]
                        y_test = data_[1][test_index]

                        model.fit(x_train, y_train)
                        result = model.predict(x_test)
                        accuracy_avg += metrics.accuracy_score(y_test, result)
                        precision_avg += metrics.precision_score(y_test, result, average='macro')
                        recall_avg += metrics.recall_score(y_test, result, average='macro')
                        f_measure_avg += metrics.f1_score(y_test, result, average='macro')

                    accuracy_avg /= 10
                    precision_avg /= 10
                    recall_avg /= 10
                    f_measure_avg /= 10

                    score = ['Dataset {}'.format(each), '{0:.2f}%'.format(accuracy_avg * 100),
                             '{0:.2f}%'.format(precision_avg * 100),
                             '{0:.2f}%'.format(recall_avg * 100), '{0:.4f}'.format(f_measure_avg)]
                    writer.writerow(score)
                '''
                file_path = 'Dataset/{}class/data.csv'.format(type_)
                data_ = pd.read_csv(file_path)

                data_ = exclude_inf(data_)

                label = 'marker'
                data_, class_number = standardizing(data_, label)

                data_.dropna(inplace=True, axis=1)
                data_ = load_data(data_, label)
                skf = StratifiedKFold(n_splits=10)
                accuracy_avg = 0
                precision_avg = 0
                recall_avg = 0
                f_measure_avg = 0

                if path == 'KMeans':
                    model = KMeans(n_clusters=class_number, random_state=0)

                for train_index, test_index in skf.split(data_[0], data_[1]):
                    x_train = data_[0][train_index]
                    y_train = data_[1][train_index]
                    x_test = data_[0][test_index]
                    y_test = data_[1][test_index]

                    model.fit(x_train, y_train)
                    result = model.predict(x_test)
                    accuracy_avg += metrics.accuracy_score(y_test, result)
                    precision_avg += metrics.precision_score(y_test, result, average='macro')
                    recall_avg += metrics.recall_score(y_test, result, average='macro')
                    f_measure_avg += metrics.f1_score(y_test, result, average='macro')

                accuracy_avg /= 10
                precision_avg /= 10
                recall_avg /= 10
                f_measure_avg /= 10

                score = ['Data', '{0:.2f}%'.format(accuracy_avg * 100),
                         '{0:.2f}%'.format(precision_avg * 100),
                         '{0:.2f}%'.format(recall_avg * 100), '{0:.4f}'.format(f_measure_avg)]
                writer.writerow(score)
