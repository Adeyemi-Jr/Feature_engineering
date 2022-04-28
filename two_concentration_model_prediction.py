import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from itertools import islice
import seaborn as sns
import os
import matplotlib.patches as mpatches
from pathlib import Path
from my_functions import  subtract_baseline_glucose, plot_glucose_concentration

date = '20220211'
glucose = ['0','300','1100','5000']

transmittance = 1
absorbance = 0
length_extension = 0
integrate = 1
detector_response = 1
two_binary_label = 0  # (300, >300)
all_window = 1 # use all data in spectrum for classification

binary_label = [[300,1100], [300,5000], [1100,5000]]


if transmittance == 1 and absorbance == 0:
    first_letter = 'T'
    output_emission_type = '_transmittance'
elif absorbance == 1 and transmittance == 0:
    first_letter = 'A'
    output_emission_type = '_absorbance'
elif (absorbance == 1 and transmittance == 1) or (absorbance == 0 and transmittance == 0):

    print('Please select one one type of data to read!')
    exit()

########### Band range

d = {'band_2': [30,65],
     'band_3': [80, 110]}
index_ = ['x_min', 'x_max']

bands_range = pd.DataFrame(data=d, index = index_ )




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


def accum_all_spectrum(LEDs):
    df = LEDs.sum(axis=1)
    return df

###########################################################################################
###########################################################################################
###########################################################################################








####################################################
####################################################
#               Pre-Processing
####################################################
####################################################

#load band, LEDs and detector response
bands = pd.read_csv('../data/processed/'+ date + '/bands_range.csv',index_col=0)
detector_resp = pd.read_csv('../data/processed/detector_response_interp.csv', index_col=0)
LEDs = pd.read_csv('../data/processed/LEDs/LED_kernels.csv')
LEDs.set_index('Wavelength', inplace=True)

#Number of measurement rounds
rounds = sorted(os.listdir('../data/raw/'+ date +'/'))

#Put all rounds into one dataframe
all_data= []
for idx, round_ in enumerate(rounds):
    appended_data_per_round_path = '../data/processed/' + date + '/' + round_ + output_emission_type+ '.csv'
    round_df = pd.read_csv(appended_data_per_round_path)
    all_data.append(round_df)

all_data = pd.concat(all_data)
all_data.reset_index(drop = True, inplace = True)


if two_binary_label == 1:
    all_data.loc[all_data[ 'glucose_level' ] > 300, 'glucose_level'] = 1100


#replace all negative value to zero, this is required for the scaling - The larger wavelengths are noisy
numeric_df = all_data.iloc[:,0:-4]
df_tmp = all_data.iloc[:,-4:]
numeric_df[numeric_df < 0] = 0



# plot LED_Kernel
fig, ax1 = plt.subplots()
ax1.plot(LEDs)
plt.legend(LEDs.columns)
plt.title('LED Spectrum (nm)')
plt.xlabel('Wavelength (nm)')
plt.show()


#plot raw data
plot_glucose_concentration(all_data, title = ' Raw data ')






#iterate over each LED spectrum, apply the LED to the data, perform subtraction from baseline, and build a accuracy map
for band in bands.columns:



    if band == 'band_1' or band == 'band_4' or band == 'band_2'  :
        continue

    elif band == 'band_2':
        LED_kernel = LEDs[['1300']].T

    elif band == 'band_3':
        LED_kernel = LEDs[['1550']].T



    if all_window:
        LED_kernel = accum_all_spectrum(LEDs)


    # apply kernel to data
    min_max_scaler = preprocessing.MinMaxScaler()
    LED_kernel = pd.DataFrame(min_max_scaler.fit_transform(LED_kernel.T))
    LED_kernel = LED_kernel.T

    #apply kernel to data
    LED_kernel_tmp = pd.concat([LED_kernel]*len(numeric_df), axis =0)
    LED_kernel_tmp.columns = numeric_df.columns
    LED_kernel_tmp.set_index(numeric_df.index, inplace = True)

    df = pd.DataFrame(numeric_df.values*LED_kernel_tmp.values)

    if detector_response:
        detector_resp.T.plot(legend = False)
        plt.title('Detector response')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Responsitivity (A/W)')

        detector_resp_tmp = pd.concat([detector_resp] * len(numeric_df), axis=0)
        detector_resp_tmp.columns = numeric_df.columns
        detector_resp_tmp.set_index(numeric_df.index, inplace=True)

        df = pd.DataFrame(df.values * detector_resp_tmp.values)


    df.columns = numeric_df.columns
    #cropped_data = pd.concat(df,all_data['glucose_level'])

    df['glucose_level'] = all_data['glucose_level']
    cropped_data = df

    if detector_response:
        plot_glucose_concentration(cropped_data, title=' Raw data after applying Kernel and Detector response ')
    else:
        plot_glucose_concentration(cropped_data, title=' Raw data after applying Kernel ')



    df = pd.concat([cropped_data, df_tmp],axis=1)

    # drop duplicate columns
    df_ = df.loc[:,~df.columns.duplicated()]



    ################################################
    ################################################
    #           Remove baseline for each rounds
    ################################################
    ################################################

    rounds = df_['Round'].unique()

    df_list = []
    for round in rounds:
        df = df_[df_['Round'] == round]

        # store df for later

        # zero glucose concentration
        df_new = df[df['glucose_level'] != 0]
        store_tmp = df_new[['Round', 'measurement_type','Temp']]
        df_new.drop(['Round', 'measurement_type','Temp'], axis=1, inplace=True)

        # non zero glucose concentration
        df_zero = df[df['glucose_level'] == 0]
        df_zero.drop(['Round', 'measurement_type', 'Temp'], axis=1, inplace=True)

        df_subtracted = subtract_baseline_glucose(df_new, df_zero)

        # df_subtracted[['Round', 'measurement_type']] = store_tmp
        result = pd.concat([df_subtracted, store_tmp.set_index(df_subtracted.index)], axis=1)

        df_list.append(result)

    final_df = pd.concat(df_list)





    ########################################################
    ########################################################
    #       plot each concentration with different color
    ########################################################
    ########################################################


    plot_glucose_concentration(final_df, title=' Post baseline subtraction ')

    glucose_level_tmp  = final_df[['glucose_level']]
    final_df_ = final_df.copy()
    for i in binary_label:

        final_df_['glucose_level'] = glucose_level_tmp
        final_df_tmPP = final_df_

        #change label to ones and zeros
        final_df_tmPP.loc[final_df_tmPP[ 'glucose_level' ] == i[0], 'glucose_level'] = 0
        final_df_tmPP.loc[final_df_tmPP['glucose_level'] == i[1], 'glucose_level'] = 1

        final_df_tmPP = final_df_tmPP[final_df_tmPP['glucose_level'] <= 1]

        #create temporary df in order to plot
        final_df_tmp  = final_df_tmPP.copy()
        final_df_tmp.loc[final_df_tmp['glucose_level'] == 0, 'glucose_level'] = i[0]
        final_df_tmp.loc[final_df_tmp['glucose_level'] == 1, 'glucose_level'] = i[1]


        y = final_df_tmPP['glucose_level']
        final_df_tmPP.drop(['Temp', 'glucose_level', 'Round', 'measurement_type'], axis=1, inplace=True)
        X = final_df_tmPP

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)




        X_test_selected_bands = []
        X_train_selected_bands = []
        column_names = []
        new_start_index = []
        window_size_tmp = []


        #iterate over the numbers of window ranges we want to look at

        if all_window == 1:
            x_1 = 0
            x_2 = 130
        else:
            x_1 = int(bands_range[band][0]) - length_extension
            x_2 = int(bands_range[band][1]) + length_extension



        length_col = list(range(x_1,x_2+1))

        #plot binary classification and set the bounds
        plot_glucose_concentration(final_df_tmp, title=' Data for binary classification ', bounds=[x_1,x_2])


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
        #list_2_df.to_csv('../results/accuracies/LEDs_accuracy_'+ band + '.csv',index= False)
        sns.heatmap(data = list_2_df, annot=True).set(title = band)
        plt.title('Accuracy score, Labels: ('+ str(i[0])+ ', ' + str(i[1])+ ') - ' + band )
        plt.show()
        A = 1
