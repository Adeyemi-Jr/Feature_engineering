import io
import os.path
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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
        df_1_tmp = pd.read_csv(df_path_1)
        df_2_tmp = pd.read_csv(df_path_2)
    else:
        #read dataframe
        df_1_tmp = df_path_1
        df_2_tmp = df_path_2


    feature_name =  list(df_1_tmp.columns)
    #get unique numbers in list
    Rounds = df_1_tmp['Round'].unique()

    output = []
    for round in Rounds:
        df_1 = df_1_tmp[df_1_tmp['Round']== round]
        df_2 = df_2_tmp[df_2_tmp['Round'] == round]
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

        tmp_features = glucose_df['Round']
        df_1.drop('Round', inplace = True, axis = 1)
        df_2.drop('Round', inplace = True, axis = 1)

        # find mean of the baseline
        basline_df_mean = baseline_df.iloc[:].mean(axis=0).to_frame().T

        #set temperature to zero
        #basline_df_mean['Temp'] = 0
        num_of_samples = len(glucose_df)

        #create template baseline df the same size as df
        baseline = pd.concat([basline_df_mean] * num_of_samples, ignore_index=True)

        glucose_df.reset_index(drop = True, inplace = True)
        baseline.reset_index(drop=True, inplace=True)
        tmp_features.reset_index(drop=True, inplace=True)

        output_df = pd.DataFrame(glucose_df.values - baseline.values)
        glucose_level = output_df.iloc[: , -1:]
        output_df = output_df.iloc[: , :-1]

        output_df['Round'] = tmp_features
        output_df['glucose_level'] = glucose_level


        #rename glucose level
        #output_df.columns = [*output_df.columns[:-1], 'glucose_level']
        #output_df['Round'] = tmp_features


        #output_df.iloc[:,0:-2].set_axis(feature_name.pop(2), axis=1, inplace=False)
        output.append(output_df)

    output = pd.concat(output)


    return output



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


#####################################################################
#####################################################################
#         function to plot glucose concentration
#####################################################################
#####################################################################

def plot_glucose_concentration(all_data, title, ignore_features = None, save = False , bounds = None, plot_type = 'line'):
    fig, ax = plt.subplots()

    color_palets_options = ['black','red','blue','green', 'orange', 'magenta']
    glucose_levels = all_data['glucose_level'].unique()


    num_concentration = len(glucose_levels)
    color_palets = color_palets_options[:num_concentration]



    glucose_level_list = []
    patches = []
    for indx, x in enumerate(glucose_levels):

        tmp = all_data[all_data['glucose_level'] == x ]
        glucose_level_list.append(tmp)


        #ignore the usless features
        if ignore_features != None:
            for ind in ignore_features:
                if ind in tmp.columns:
                    tmp.drop(ind, inplace = True, axis =1)

        tmp.drop('glucose_level', inplace = True, axis = 1)
        colour = color_palets[indx]
        #ax.plot(tmp.columns, tmp.T, color=colour, kind=plot_type)

        if plot_type == 'line':
            ax.plot(tmp.columns, tmp.T, color=colour)
        elif plot_type == 'scatter':
            ax.plot(tmp.columns, tmp.T, color=colour, marker= 'o',markerfacecolor='None', linestyle = 'None')

        patch = mpatches.Patch(color=colour, label='Glucose Level ' + str(x))
        patches.append(patch)


    num_x_tick = round(len(tmp.columns)/10)
    if num_x_tick == 0:
        num_x_tick = 4


    ax.xaxis.set_major_locator(plt.MaxNLocator(num_x_tick))
    plt.legend(handles=patches)
    plt.xlabel('Wavelength')
    plt.grid()
    plt.title(title)


    if bounds is not None:
        x_1 = bounds[0]
        x_2 = bounds[1]
        ax.axvspan(x_1, x_2, alpha=0.3, color='red')



    plt.show()


def find_nearest(array, values, K=1):

    '''
    Returns the index of the nearest two element to the value
    :param array:
    :param values:
    :return: indices
    '''
    #indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    #return indices

    #K = 2
    X = abs(array - values)
    indexes = sorted(range(len(X)), key=lambda sub: X[sub])[:K]
    return indexes