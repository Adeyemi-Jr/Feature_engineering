import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from my_functions import read_NIR_data
import os

from sklearn import preprocessing



#Import data

raw_data_path = '../data/raw/LEDs/20220222_RI/'
peaks = [name for name in os.listdir(raw_data_path) if os.path.isdir(name)]

#count the numbers of txt files
#peaks = ['','','','','','']
LED_peaks = ['1300','1450','1550','1650a','1650b']
#fileCounter = len(glob.glob1(raw_data_path+ str(peaks[0]) , '*.txt'))

#raw_data_path_txt_file = raw_data_path + LED_peaks[0] + '/RI_' + LED_peaks[0]  +'_1.txt'

#iterate over peaks
data = []
for peak in LED_peaks:

    #list_of_txt_file = [name for name in os.listdir(raw_data_path + txtfile) if os.path.isdir(name)]

    filedestination = raw_data_path+peak
    text_files = [f for f in os.listdir(filedestination) if f.endswith('.txt')]
    text_files= sorted(text_files)
    new_df = []
    for i, txtfile in enumerate(text_files) :
        index = i+1
        raw_data_path_txt_file = raw_data_path + peak + '/' + txtfile
        df, temp = read_NIR_data(raw_data_path_txt_file, is_transmittance = True)
        wavelength = df['Wavelength']
        df.drop('Wavelength', axis = 1, inplace=True)
        new_df.append(df)

    new_df = pd.concat(new_df, axis = 1)


    #remove dublicate column
    new_df_ = new_df.mean(axis = 1)
    new_df = new_df_.to_frame()
    new_df['Wavelength'] = wavelength
    new_df.set_index('Wavelength',inplace=True)
    new_df.columns = [peak]
    #new_df.plot()
    #plt.show
    data.append(new_df)

data = pd.concat(data, axis = 1)

#compute average of the two 1650 wavelengths
df_tmp = data[['1650a','1650b']]

df_tmp = df_tmp.mean(axis=1)
df_tmp = df_tmp.to_frame()

data.drop(['1650a','1650b'], axis = 1, inplace=True)
data['1650'] = df_tmp



min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(data[['1300']])

scaled_data = []
for i in data.columns:
    tmp = pd.DataFrame(min_max_scaler.transform(data[[i]]))
    scaled_data.append(tmp)

scaled_data = pd.concat(scaled_data,axis = 1)
scaled_data.columns = data.columns


plt.plot(scaled_data)

data.to_csv('../data/processed/LEDs/LED_kernels.csv')
scaled_data.to_csv('../data/processed/LEDs/LED_kernels_scaled.csv')
