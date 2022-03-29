import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from itertools import islice
import seaborn as sns


date = '20220211'
integrate = 1
extend_band = 1
LED_present = 1


processed_cropped_path = '../data/processed/'+ date + '/cropped_data.csv'
processed_cropped = pd.read_csv(processed_cropped_path)

bands_range_path = '../data/processed/'+ date + '/bands_range.csv'
bands_range = pd.read_csv(bands_range_path,index_col=0)


if LED_present == 1:
    LED_path = '../data/processed/LEDs/LED_kernels.csv'
    LED_spectrum = pd.read_csv(LED_path)


#encode the label column
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 300] = 0
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 1100] = 1
processed_cropped['glucose_level'][processed_cropped['glucose_level'] == 5000] = 2

'''
# Plot data, bands and the spectrum
if LED_present == 1:
    df_5000_glucose = processed_cropped[processed_cropped['glucose_level'] == 2]
    df_5000_glucose.drop(['Temp', 'glucose_level', 'Round', 'measurement_type'], axis=1, inplace=True)
    df_5000_glucose = df_5000_glucose.mean(axis=0).to_frame()
    peaks = bands_range.iloc[0].values.tolist()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df_5000_glucose.iloc[10:-47],  'g-', label = 'data')
    ax2.plot(LED_spectrum.iloc[10:-47, 1:], 'b-', label = 'LED')

    for i in range(0, len(peaks)):
        x_1 = bands_range.iloc[1, i]-10
        x_2 = bands_range.iloc[2, i]-10
        ax1.axvspan(x_1, x_2, alpha=0.5, color='red')
    plt.show()

'''




y = processed_cropped['glucose_level']
processed_cropped.drop(['Temp','glucose_level','Round','measurement_type'],axis = 1, inplace = True)
X = processed_cropped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


if extend_band:
    length_extension = 5
else:
    length_extension = 0

#############################################################
#############################################################
# Step 1
#############################################################
#############################################################


model_1 = RandomForestClassifier(random_state=42)
model_1.fit(X_train,y_train)

feat_importances = pd.Series(model_1.feature_importances_, index=processed_cropped.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

y_hat = model_1.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
A = 1




'''
plt.plot(X.T)
plt.show()
'''



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

    model = RandomForestClassifier(random_state = 42)
    #model = SVC(random_state=42)
    model.fit(X_train_cropped, y_train_cropped)
    yhat = model.predict(X_test_cropped)
    acc = accuracy_score(y_test_cropped, yhat)

    return acc

###########################################################################################
###########################################################################################
###########################################################################################

X_test_selected_bands = []
X_train_selected_bands = []
column_names = []
new_start_index = []
window_size_tmp = []

#iterate over the numbers of window ranges we want to look at
for idx, col in enumerate(bands_range):

    x_1 = int(bands_range[col].iloc[1]) - length_extension
    x_2 = int(bands_range[col].iloc[-1]) + length_extension
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

            if integrate  == 1:
                X_train_cropped_ = X_train_cropped.sum(axis=1)
                X_train_cropped = X_train_cropped_.to_frame()

                X_test_cropped_ = X_test_cropped.sum(axis=1)
                X_test_cropped =  X_test_cropped_.to_frame()

            #Apply model to the cropped df and get accuracy score
            acc = create_model_and_accuracy(X_train_cropped, y_train,X_test_cropped, y_test)
            accuracy_per_window_list.append(acc)
        accuracy.append(accuracy_per_window_list)

    list_2_df = pd.DataFrame(accuracy, columns=length_col)
    list_2_df.to_csv('../results/accuracies/accuracy_'+ col + '.csv',index= False)
    sns.heatmap(data = list_2_df, annot=True).set(title = col)
    #plt.title(col, annot=True)

    if not integrate:

        # select the chosen features per branch
        if col == 'band_1':

            starting_index = 12
            window_size = 3
            df_range_train = X_train.iloc[:,starting_index : starting_index + window_size]
            df_range_test = X_test.iloc[:,starting_index : starting_index + window_size]

        elif col == 'band_2':

            starting_index = 46
            window_size = 4
            df_range_train = X_train.iloc[:,starting_index : starting_index + window_size]
            df_range_test = X_test.iloc[:,starting_index : starting_index + window_size]

        elif col == 'band_3':

            starting_index = 86
            window_size = 5
            df_range_train = X_train.iloc[:,starting_index : starting_index + window_size]
            df_range_test = X_test.iloc[:,starting_index : starting_index + window_size]
        elif col == 'band_4':

            starting_index = 124
            window_size = 4
            df_range_train = X_train.iloc[:, starting_index: starting_index + window_size]
            df_range_test = X_test.iloc[:, starting_index: starting_index + window_size]

    elif integrate:
        # select the chosen features per branch
        if col == 'band_1':

            starting_index = 22
            window_size = 3

            df_range_train = X_train.iloc[:, starting_index: starting_index + window_size]
            df_range_test = X_test.iloc[:, starting_index: starting_index + window_size]

            # project training and test data into 1D
            df_range_train_ = df_range_train.sum(axis=1)
            df_range_train = df_range_train_.to_frame()

            df_range_test_ = df_range_test.sum(axis=1)
            df_range_test = df_range_test_.to_frame()

        elif col == 'band_2':

            starting_index = 55
            window_size = 3
            df_range_train = X_train.iloc[:, starting_index: starting_index + window_size]
            df_range_test = X_test.iloc[:, starting_index: starting_index + window_size]

            # project training and test data into 1D
            df_range_train_ = df_range_train.sum(axis=1)
            df_range_train = df_range_train_.to_frame()

            df_range_test_ = df_range_test.sum(axis=1)
            df_range_test = df_range_test_.to_frame()

        elif col == 'band_3':

            starting_index = 96
            window_size = 5
            df_range_train = X_train.iloc[:, starting_index: starting_index + window_size]
            df_range_test = X_test.iloc[:, starting_index: starting_index + window_size]

            # project training and test data into 1D
            df_range_train_ = df_range_train.sum(axis=1)
            df_range_train = df_range_train_.to_frame()

            df_range_test_ = df_range_test.sum(axis=1)
            df_range_test = df_range_test_.to_frame()

        elif col == 'band_4':

            starting_index = 133
            window_size = 4
            df_range_train = X_train.iloc[:, starting_index: starting_index + window_size]
            df_range_test = X_test.iloc[:, starting_index: starting_index + window_size]

            #project training and test data into 1D
            df_range_train_ = df_range_train.sum(axis=1)
            df_range_train = df_range_train_.to_frame()

            df_range_test_ = df_range_test.sum(axis=1)
            df_range_test = df_range_test_.to_frame()


    column_names.append(col)
    new_start_index.append(starting_index)
    window_size_tmp.append(window_size)

    X_test_selected_bands.append(df_range_test)
    X_train_selected_bands.append(df_range_train)


df_range= pd.DataFrame({
        "start_index":new_start_index,
        "window_size": window_size_tmp
        })

#store selected index



X_test_selected_bands_df  = pd.concat(X_test_selected_bands,axis=1)
X_train_selected_bands_df = pd.concat(X_train_selected_bands,axis=1)











#final
acc = create_model_and_accuracy(X_train_selected_bands_df, y_train,X_test_selected_bands_df, y_test)
print('Accuracy:', acc)





'''
data = d = {'Band 1': [0.81], 'Band 2': [0.98], 'Band 3': [0.89], 'Band 1,2': [0.98], 'Band 1,2,3': [1] }
Final_score = pd.DataFrame(d)
plt.plot(Final_score.T, marker = 'o')
plt.show()
plt.xlabel('Bands')
plt.ylabel(' Accuracy ')
A = 1
'''


fig, ax = plt.subplots()
ax.plot(X_train.T)

for i in range(0,len(df_range)):
    x_1 = df_range.iloc[i,0]
    x_2 = x_1 + df_range.iloc[i,1]
    ax.axvspan(x_1, x_2, alpha=0.5, color='red')

plt.xlabel('wavelength')
plt.ylabel('transmittance')



if integrate:
    plt.title('Glucose concentration (Accumulation)')
else:
    plt.title('Glucose concentration')

plt.show()
A=1



