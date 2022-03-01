import numpy as np
import pandas as pd
from my_functions import subtract_baseline_glucose
from matplotlib import pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths


date = '20220211'
processed_data_path = '../data/processed/'+ date +'/processed.csv'
processed_data  = pd.read_csv(processed_data_path)



################################################
################################################
#           Remove baseline for each rounds
################################################
################################################

rounds = processed_data['Round'].unique()


df_list = []
for round in rounds:

    df = processed_data[processed_data['Round'] == round]

    #store df for later

    #zero glucose concentration
    df_new = df[df['glucose_level'] != 0]
    store_tmp = df_new[['Round', 'measurement_type']]
    df_new.drop(['Round', 'measurement_type'], axis=1, inplace=True)


    # non zero glucose concentration
    df_zero =  df[df['glucose_level'] == 0]
    df_zero.drop(['Round', 'measurement_type'], axis=1, inplace=True)

    df_subtracted = subtract_baseline_glucose(df_new, df_zero )

    #df_subtracted[['Round', 'measurement_type']] = store_tmp
    result = pd.concat([df_subtracted, store_tmp.set_index(df_subtracted.index)], axis=1)

    df_list.append(result)

final_df = pd.concat(df_list)


###################################################
###################################################
#                      FInd peaks and FWHM
###################################################
###################################################



#use glucose concentreation of 1100 or 5000, lower glucose values are more nosier
glucose_level = 5000
new_df = final_df[final_df['glucose_level'] == glucose_level]
new_df.drop(['Temp', 'glucose_level', 'measurement_type', 'Round'],axis = 1, inplace = True)
new_df = new_df.T


new_df.plot(legend = False)
plt.title('glucose level '+ str(glucose_level) +' mg/dl')
plt.xlabel('Wavelength (nm)')




# Number of rows to drop
n = abs (190 - len(new_df))

#Remove the last n rows
df_dropping_last_n = new_df.iloc[10:-n]

df_dropping_last_n.plot(legend = False)
plt.title('glucose level '+ str(glucose_level) +' mg/dl (zoomed)')
plt.xlabel('Wavelength (nm)')
plt.show()


#Compute mean of all measuremnts
df2 = df_dropping_last_n.mean(axis=1).to_frame()
df2.plot()
plt.title('glucose level '+ str(glucose_level) +' mg/dl (mean)')
plt.xlabel('Wavelength (nm)')
plt.show()









column_list = df2.index.values.tolist()
df_2_array = df2.to_numpy().flatten()
x = df_2_array

peaks, _ = find_peaks(df_2_array, prominence=1)
peaks = [14,51,125]
fwhm_approx = peak_widths(df_2_array, peaks, rel_height=0.6)


plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['peaks'])
plt.show()


plt.plot(x)

plt.plot(peaks, x[peaks], "x")

plt.hlines(*fwhm_approx[1:], color="C2")
plt.show()



d = {'peak 1': [peaks[0], np.round(fwhm_approx[2][0]), np.round(fwhm_approx[3][0]) ],
     'peak 2': [peaks[1], np.round(fwhm_approx[2][1]), np.round(fwhm_approx[3][1]) ],
     'peak 3': [peaks[2], np.round(fwhm_approx[2][2]), np.round(fwhm_approx[3][2]) ]}

index_ = ['peak_point', 'x_min', 'x_max']
df_range = pd.DataFrame(data=d,index=index_)






'''
final_df_.reset_index(drop  = True, inplace = True)



# Number of rows to drop
n = abs (190 - len(final_df_))

#Remove the last n rows
df_dropping_last_n = final_df_.iloc[:-n]

df_dropping_last_n.plot()
plt.show()

A=1
'''