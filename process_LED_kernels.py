import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from my_functions import read_NIR_data
import os



#Import data

raw_data_path = '../data/raw/LEDs/20220222_RI/'
peaks = [name for name in os.listdir(raw_data_path) if os.path.isdir(name)]

#count the numbers of txt files
#peaks = ['','','','','','']
LED_peaks = ['1300']
#fileCounter = len(glob.glob1(raw_data_path+ str(peaks[0]) , '*.txt'))

#raw_data_path_txt_file = raw_data_path + LED_peaks[0] + '/RI_' + LED_peaks[0]  +'_1.txt'

#iterate over peaks
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
    new_df.plot()
    plt.show

A = 1