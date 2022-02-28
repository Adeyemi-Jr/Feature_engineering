import io
import os.path
import numpy as np
import pandas as pd

def read_RF_data(path,S_parameter):

    #print(path)

    data_string = open(path).read()

    ignore_lines = 10
    New_string = data_string.split('\n', ignore_lines)[ignore_lines]
    data_tmp = io.StringIO(New_string)
    df = pd.read_csv(data_tmp, names = ['FREQ.GHZ','S11DB','S11A','S21DB','S21A','S12DB','S12A','S22DB','S22A'], delim_whitespace=True)
    new_df = df[['FREQ.GHZ',S_parameter]]

    return new_df


# Absorbance or Transmitance




def read_NIR_data(path, is_transmittance = True):

    if is_transmittance == True:
        file_name = "T0"
        output_path = "/Transmittance.csv"
    else:
        file_name = "A0"
        output_path = "/Absorbance.csv"

    data_string = open(path).read().replace(';', ',')
    temp = data_string.split("\n")[0]


    # ignore the first 7 lines since they dont contain useful information
    ignore_lines = 7
    New_string = data_string.split("\n", ignore_lines)[ignore_lines]
    data_tmp = io.StringIO(New_string)
    df = pd.read_csv(data_tmp, names=['Wavelength', 'Sample', 'Dark', 'Reference', 'Transmittance'])
    new_df = df[['Wavelength', 'Transmittance']]
    return new_df, temp






def subtract_baseline_glucose(df_path_1, df_path_2 ):

    #check if input is a path or a dataframe
    if type(df_path_1) == str:
        #read path
        df_1 = pd.read_csv(df_path_1)
        df_2 = pd.read_csv(df_path_2)
    else:
        #read dataframe
        df_1 = df_path_1
        df_2 = df_path_2

    # assign dataframe to dictionary
    df_dict = {}
    df_dict[1] = df_1
    df_dict[2] = df_2

    # find which dataframe has zero glucose
    if df_dict[1]['glucose_level'].sum() == 0:
        baseline_df = df_1
        glucose_df = df_2

    elif df_dict[2]['glucose_level'].sum() == 0:
        baseline_df = df_2
        glucose_df = df_1

    # find mean of the baseline
    basline_df_mean = baseline_df.iloc[:].mean(axis=0).to_frame().T

    #set temperature to zero
    basline_df_mean['Temp'] = 0
    num_of_samples = len(glucose_df)

    #create template baseline df the same size as df
    baseline = pd.concat([basline_df_mean] * num_of_samples, ignore_index=True)

    glucose_df.reset_index(drop = True, inplace = True)
    baseline.reset_index(drop=True, inplace=True)
    output_df = glucose_df.subtract(baseline)

    return output_df



def remove_temp_and_glucose_and_transpose(df, transpose = False):

    '''
    removes the temperature and glucose column, has the option to perform transpose of the dataframe.
    This function is useful when plotting

    :param df:
    :return new_df:
    '''
    new_df = df.drop(['Temp', 'glucose_level'], axis=1)

    if transpose == True:

        new_df = new_df.T

        return new_df
    else:
        return new_df



def Absorbance_2_Transmittance(input, output_type):

    '''

    :param input: Input dataframe either transmission or absorbance
    :param output_type: either "T" or "A"
    :return: return the transformed dataframe
    '''


    if output_type == 'T':
       output = 10**(2-input)


    elif output_type == 'A':
        output = np.log10(input)



    return output


