import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import islice


date = '20220211'


processed_cropped_path = '../data/processed/'+ date + '/cropped_data.csv'
processed_cropped = pd.read_csv(processed_cropped_path)

bands_range_path = '../data/processed/'+ date + '/bands_range.csv'
bands_range = pd.read_csv(bands_range_path,index_col=0)

#encode the label column
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 300] = 0
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 1100] = 1
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 5000] = 2

y = processed_cropped['glucose_level']
processed_cropped.drop(['Temp','glucose_level','Round','measurement_type'],axis = 1, inplace = True)
X = processed_cropped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)








###########################################################################################
###########################################################################################
###########################################################################################


def sliding_window(seq, window_size=2):
    "Returns a sliding window (of width n) over data from the iterable"
    window_list = []
    for i in range(len(seq) - window_size + 1):
        window_list.append(seq[i: i + window_size])
    return window_list


def create_model_and_accuracy(X_train_cropped, y_train_cropped,X_test_cropped, y_test_cropped):
    '''

    :param X_train_cropped:
    :param y_train_cropped:
    :param X_test_cropped:
    :param y_test_cropped:
    :return:
    '''

    model = RandomForestClassifier()
    model.fit(X_train_cropped, y_train_cropped)
    yhat = model.predict(X_test_cropped)
    acc = accuracy_score(y_test_cropped, yhat)

    return acc

###########################################################################################
###########################################################################################
###########################################################################################

#iterate over the numbers of window ranges we want to look at
for idx, col in enumerate(bands_range):

    x_1 = int(bands_range[col].iloc[1])
    x_2 = int(bands_range[col].iloc[-1])
    length_col = list(range(x_1,x_2+1))

    accuracy = []
    #iterate over the length of the window
    for window_size in range(0,len(length_col)):
        windows = sliding_window(length_col, window_size+1)

        accuracy_per_window_list = []
        #iterate over the range of windows 1's, 2's ... etc
        for w in windows:
            X_train_cropped = X_train.iloc[:,w[0]:w[-1]+1]
            X_test_cropped = X_test.iloc[:, w[0]:w[-1] + 1]


            #Apply model to the cropped df and get accuracy score
            acc = create_model_and_accuracy(X_train_cropped, y_train,X_test_cropped, y_test)
            accuracy_per_window_list.append(acc)
        accuracy.append(accuracy_per_window_list)

    list_2_df = pd.DataFrame(accuracy, columns=length_col)
    list_2_df.to_csv('../results/accuracies/accuracy_'+ col + '.csv')