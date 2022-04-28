import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from my_functions import subtract_baseline_glucose
from my_functions import plot_glucose_concentration




date = '20220211'
glucose = ['0','300','1100','5000']
output_emission_type = '_transmittance'


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



def ingetrage_all_frequency(data, LED_kernel_input, detector_response,list_tmp_features = ['Temp', 'Round','measurement_type', 'glucose_level']):

    data = data.copy()
    #plot original data
    plot_glucose_concentration(data, title = 'Raw Data', ignore_features=['Temp', 'measurement_type', 'Round'], save=False,bounds=None)

    tmp_df = data[list_tmp_features]
    data.drop(list_tmp_features,axis = 1, inplace = True)
    numeric_df = data
    num_of_samples = len(data)


    # apply kernel to data
    kernel_name = list(LED_kernel_input.columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    LED_kernel = pd.DataFrame(min_max_scaler.fit_transform(LED_kernel_input))
    LED_kernel.set_index(LED_kernel_input.index, inplace = True)
    LED_kernel.rename(columns={LED_kernel.columns[0]: kernel_name[0]}, inplace=True)


    #plot kernel
    LED_kernel.plot()
    plt.title(kernel_name[0]+ ' nm')
    plt.xlabel('Wavelength')
    plt.grid()
    plt.show()

    LED_kernel = LED_kernel.T
    LED_kernel_ = pd.concat([LED_kernel] * num_of_samples, axis=0)

    df = pd.DataFrame(numeric_df.values * LED_kernel_.values)

    #apply detector response
    detector_resp_tmp = pd.concat([detector_response] * num_of_samples, axis=0)
    detector_resp_tmp.columns = numeric_df.columns
    detector_resp_tmp.set_index(numeric_df.index, inplace=True)


    df = pd.DataFrame(df.values * detector_resp_tmp.values)

    df.columns = numeric_df.columns

    #add Glucose concentration and Rounds to df
    df[['Round','glucose_level']] = tmp_df[['Round','glucose_level']]


    #subtract baseline measurment

    # non-zero glucose concentration
    df_new = df[df['glucose_level'] != 0]

    # zero glucose concentration
    df_zero = df[df['glucose_level'] == 0]

    df_subtracted = subtract_baseline_glucose(df_new, df_zero)
    df_subtracted.reset_index(drop=True, inplace=True)


    df_subtracted.set_axis(df_new.columns, axis=1, inplace=True)

    #df_subtracted['glucose_level', 'Round'] = df_subtracted_tmp['glucose_level', 'Round']



    plot_glucose_concentration(df_subtracted, title = 'Deltas after applying Kernel ('+ kernel_name[0] +'nm) and detector response', ignore_features=['Round'], save=False,bounds=None)

    output_df = df_subtracted.iloc[:,0:-2]
    output_df_ = output_df.sum(axis=1).to_frame()
    output = output_df_
    output['glucose_level'] = df_subtracted['glucose_level']

    A = 1

    return output




# plot detector response
detector_resp.T.plot()
plt.title('Detector response')
plt.xlabel('Wavelength')
plt.grid()
plt.show()


num_LEDs = LEDs.columns
integrated_df = []
for Led in num_LEDs:

    Led_ = LEDs[[Led]]
    integrate_ = ingetrage_all_frequency(all_data, Led_, detector_resp)



    integrate_.rename(columns={ integrate_.columns[0]: Led+"nm" }, inplace = True)

    integrated_df.append(integrate_)

integrated_df = pd.concat(integrated_df)



A = 1