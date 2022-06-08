import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from my_functions import subtract_baseline_glucose
from my_functions import plot_glucose_concentration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from my_functions import find_nearest_2
import seaborn as sns



def sliding_window_mid_point(df, index_LED, size):

    mask_array = np.zeros(df.shape)
    x = index_LED-size
    y = index_LED+size+1

    mask_array[:,x:y] = 1.0

    mask_df = pd.DataFrame(mask_array)
    output_df = df.values*mask_df.values
   #output_df.rename(columns = df.columns, inplace = True)
    output = pd.DataFrame(data = output_df, columns=df.columns)
    return output













LED_wavelenghts = ['1300','1450','1550','1650']


LED_combination  = ['1300','1450', '1550' ,'1650']

df_1300 = pd.read_csv('../data/processed/20220211/processed_delta_'+ LED_wavelenghts[0]  +'nm.csv')
df_1450 = pd.read_csv('../data/processed/20220211/processed_delta_'+ LED_wavelenghts[1]  +'nm.csv')
df_1550 = pd.read_csv('../data/processed/20220211/processed_delta_'+ LED_wavelenghts[2]  +'nm.csv')
df_1650 = pd.read_csv('../data/processed/20220211/processed_delta_'+ LED_wavelenghts[3]  +'nm.csv')


#encode the label columns
df_1300['glucose_level'][df_1300['glucose_level'] == 300] = 0
df_1300['glucose_level'][df_1300['glucose_level'] == 1100] = 1
df_1300['glucose_level'][df_1300['glucose_level'] == 5000] = 2

df_1450['glucose_level'][df_1450['glucose_level'] == 300] = 0
df_1450['glucose_level'][df_1450['glucose_level'] == 1100] = 1
df_1450['glucose_level'][df_1450['glucose_level'] == 5000] = 2

df_1550['glucose_level'][df_1550['glucose_level'] == 300] = 0
df_1550['glucose_level'][df_1550['glucose_level'] == 1100] = 1
df_1550['glucose_level'][df_1550['glucose_level'] == 5000] = 2

df_1650['glucose_level'][df_1650['glucose_level'] == 300] = 0
df_1650['glucose_level'][df_1650['glucose_level'] == 1100] = 1
df_1650['glucose_level'][df_1650['glucose_level'] == 5000] = 2






wavelenghts = np.asarray(df_1300.columns[:-1], dtype=float)

#find the min and max value of the LEDs availiable
max_wavelength = int(max(LED_wavelenghts))
min_wavelength = int(min(LED_wavelenghts))

#find the min and max index of the LEDs avaialble
min_index_LED, value_1=  find_nearest_2(wavelenghts, min_wavelength)
max_index_LED, value_2 = find_nearest_2(wavelenghts, max_wavelength)


window_size_limit = min(min_index_LED, abs(max_index_LED- len(wavelenghts)))
window_size=0



index_LED_1300, _ = find_nearest_2(wavelenghts, int(LED_wavelenghts[0]))
index_LED_1450, _ = find_nearest_2(wavelenghts, int(LED_wavelenghts[1]))
index_LED_1550, _ = find_nearest_2(wavelenghts, int(LED_wavelenghts[2]))
index_LED_1650 ,_ = find_nearest_2(wavelenghts, int(LED_wavelenghts[3]))


y =  df_1300['glucose_level']
df_1300_ = df_1300.iloc[:,:-1]
df_1450_ = df_1450.iloc[:,:-1]
df_1550_ = df_1550.iloc[:,:-1]
df_1650_ = df_1650.iloc[:,:-1]


## Create sliding wondow from mid point
accuracy_list = []
while window_size <= window_size_limit:

    #apply mask to df
    df_1300 = sliding_window_mid_point(df_1300_, index_LED_1300, window_size)
    df_1450 = sliding_window_mid_point(df_1450_, index_LED_1450, window_size)
    df_1550 = sliding_window_mid_point(df_1550_, index_LED_1450, window_size)
    df_1650 = sliding_window_mid_point(df_1650_, index_LED_1650, window_size)

    #integrate df one column
    df_1300 = pd.DataFrame(df_1300.sum(axis=1))
    df_1450 = pd.DataFrame(df_1450.sum(axis=1))
    df_1550 = pd.DataFrame(df_1550.sum(axis=1))
    df_1650 = pd.DataFrame(df_1650.sum(axis=1))

    #Change the column name 
    df_1300.rename(columns={ df_1300.columns[0]: '1300' }, inplace = True)
    df_1450.rename(columns={ df_1450.columns[0]: '1450' }, inplace = True)
    df_1550.rename(columns={df_1550.columns[0]: '1550'}, inplace=True)
    df_1650.rename(columns={ df_1650.columns[0]: '1650' }, inplace = True)


    X_tmp = pd.concat([df_1300, df_1450, df_1550, df_1650], axis=1)

    X = X_tmp[LED_combination]
    #Data split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    y_hat = model.predict(X_test)

    acc = accuracy_score(y_test, y_hat)

    accuracy_list.append(acc)
    window_size+=1








    print('Window size:', window_size, '    Acc:', acc )
A  =1
plt.plot(accuracy_list, marker= '*',color = 'r')
plt.title('LED combination:' + str(LED_combination), fontsize=18)
plt.xlabel('window size')
plt.ylabel('Accuracy')
plt.ylim(0.94,1)

plt.show()







#if __name__ == "__main__":



















''' 
integrated_df = pd.read_csv('../data/processed/20220211/processed_integrated.csv')


#encode the label column
integrated_df['glucose_level'][integrated_df['glucose_level'] == 300] = 0
integrated_df['glucose_level'][integrated_df['glucose_level'] == 1100] = 1
integrated_df['glucose_level'][integrated_df['glucose_level'] == 5000] = 2


y = integrated_df['glucose_level']
X = integrated_df
X.drop(['glucose_level'], axis = 1, inplace = True)
#X.drop(['glucose_level'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


model = RandomForestClassifier(random_state=42)
# model = SVC(random_state=42)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
acc = accuracy_score(y_test, yhat)
print('Accuracy: ', acc)

'''