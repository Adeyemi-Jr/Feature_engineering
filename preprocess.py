import numpy as np
import pandas as pd
from my_functions import read_NIR_data
import os
from pathlib import Path
from my_functions import Absorbance_2_Transmittance

date = '20220211'
glucose = ['0','300','1100','5000']

transmittance = 1
absorbance = 0


if transmittance == 1 and absorbance == 0:
    first_letter = 'T'
    output_emission_type = '_transmittance'
elif absorbance == 1 and transmittance == 0:
    first_letter = 'A'
    output_emission_type = '_absorbance'
elif (absorbance == 1 and transmittance == 1) or (absorbance == 0 and transmittance == 0):

    print('Please select one one type of data to read!')
    exit()

rounds = sorted(os.listdir('../data/raw/'+ date +'/'))

for indx_round, round in enumerate(rounds):
    path_dir = '../data/raw/' + date + '/' + round
    appended_data_per_round = []

    for glucose_concentration in glucose:
        path_glucose_ = path_dir + '/' + glucose_concentration + '/'

        # get numbers of transmittance files
        num_trans_file = []
        for file in os.listdir(path_glucose_):
            if file.startswith(first_letter):
                num_trans_file.append(file)

        num_trans_file.sort()

        # append data for each glucose concentration (loop)
        append_data = []
        for file_name in num_trans_file:
            path = path_glucose_ + file_name

            # read NIR file
            df, temp = read_NIR_data(path)
            df_transposed = df.transpose()

            # Use first row as header- wavelengths
            new_header = df_transposed.iloc[0]
            df_transposed = df_transposed[1:]
            df_transposed.columns = new_header
            df_transposed.reset_index(drop=True, inplace=True)

            # add temperature, round number, and glucose reading to dataframe
            df_transposed['Temp'] = temp
            df_transposed['Round'] = indx_round + 1
            df_transposed['measurement_type'] = first_letter
            df_transposed['glucose_level'] = glucose_concentration
            append_data.append(df_transposed)

        # put all measurments corresponding to same glucose reading into df
        appended_data = pd.concat(append_data)
        A = 1
        # append the df for different glucose concentration level
        appended_data_per_round.append(appended_data)

    appended_data_per_round = pd.concat(appended_data_per_round)
    appended_data_per_round_path = '../data/processed/' + date + '/' + round + output_emission_type + '.csv'
    appended_data_per_round.to_csv(appended_data_per_round_path, index=False)





#Put all rounds into one dataframe
all_data= []
for idx, round_ in enumerate(rounds):
    appended_data_per_round_path = '../data/processed/' + date + '/' + round_ + output_emission_type+ '.csv'
    round_df = pd.read_csv(appended_data_per_round_path)
    all_data.append(round_df)

all_data = pd.concat(all_data)
all_data.reset_index(drop = True, inplace = True)
all_data.to_csv('../data/processed/'+ date + '/processed.csv',index = False)
