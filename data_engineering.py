import numpy as np
import pandas as pd
from my_functions import subtract_baseline_glucose
from matplotlib import pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
import matplotlib.patches as mpatches


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



# plot each concentration with different color
df_300 = final_df[final_df['glucose_level'] == 300]
df_300.drop(['glucose_level', 'Round', 'measurement_type'],axis = 1, inplace = True)
df_300 = df_300.T

df_1100 = final_df[final_df['glucose_level'] == 1100]
df_1100.drop(['glucose_level', 'Round', 'measurement_type'],axis = 1, inplace = True)
df_1100 = df_1100.T

df_5000 = final_df[final_df['glucose_level'] == 5000]
df_5000.drop(['glucose_level', 'Round', 'measurement_type'],axis = 1, inplace = True)
df_5000 = df_5000.T

fig = plt.figure(figsize = (20,10))
plt.plot(df_300, color = 'red', label = 'glucose_level 300')
plt.plot(df_1100, color = 'blue', label = 'glucose_level 1000')
plt.plot(df_5000, color = 'green', label = 'glucose_level 5000')


red_patch = mpatches.Patch(color='red',label = 'glucose_level 300')
blue_patch = mpatches.Patch(color='blue',label = 'glucose_level 1100')
green_patch = mpatches.Patch(color='green',label = 'glucose_level 5000')
plt.legend(handles=[red_patch,blue_patch,green_patch])

plt.title('Glucose concentration')
plt.xlabel('wavelength')
plt.ylabel('transmittance')
plt.show()

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
#df_dropping_last_n = new_df

df_dropping_last_n.plot(legend = False)
plt.title('glucose level '+ str(glucose_level) +' mg/dl (cropped)')
plt.xlabel('Wavelength (nm)')
plt.show()


#Compute mean of all measuremnts
#df2 = df_dropping_last_n.mean(axis=1).to_frame()
df2 = new_df.mean(axis=1).to_frame()
df2.iloc[10:-n].plot()
plt.title('glucose level '+ str(glucose_level) +' mg/dl (mean)')
plt.xlabel('Wavelength (nm)')
plt.show()









column_list = df2.index.values.tolist()
df_2_array = df2.to_numpy().flatten()
x = df_2_array

peaks, _ = find_peaks(df_2_array, prominence=1)
peaks = [14, 51, 95, 125]
peaks = [x+10 for x in peaks]
fwhm_approx = peak_widths(df_2_array[10:-n], peaks, rel_height=0.6)


plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['peaks'])
plt.show()


plt.plot(x)

plt.plot(peaks, x[peaks], "x")

plt.hlines(*fwhm_approx[1:], color="C2")
plt.show()


#manually add the 3rd peak and index
'''
d = {'band_1': [peaks[0], np.round(fwhm_approx[2][0]), np.round(fwhm_approx[3][0]) ],
     'band_2': [peaks[1], np.round(fwhm_approx[2][1]), np.round(fwhm_approx[3][1]) ],
     'band_3': [95, 78, 111 ],
     #'band_4': [peaks[3], np.round(fwhm_approx[2][3]), np.round(fwhm_approx[3][3]) ]
     'band_4': [peaks[3], np.round(fwhm_approx[2][3]), 145 ]}

d = {'band_1': [peaks[0], np.round(fwhm_approx[2][0]), np.round(fwhm_approx[3][0]) ],
     'band_2': [peaks[1], np.round(fwhm_approx[2][1]), np.round(fwhm_approx[3][1]) ],
     'band_3': [peaks[2], np.round(fwhm_approx[2][2]), np.round(fwhm_approx[3][2]) ]}
'''
#manually selection of the bands
d = {'band_1': [ peaks[0], 20, 33 ],
     'band_2': [ peaks[1], 48, 66 ],
     'band_3': [ peaks[2], 88, 121 ],
     'band_4': [ peaks[3], 127, 140 ]}

index_ = ['peak_point', 'x_min', 'x_max']




## save outputs
df_range = pd.DataFrame(data=d,index=index_)
df_range.to_csv('../data/processed/'+ date + '/bands_range.csv')

df_tmp = final_df[['Temp', 'glucose_level', 'measurement_type', 'Round']]
final_df.drop(['Temp', 'glucose_level', 'measurement_type', 'Round'], inplace  = True, axis = 1)
#cropped_data = final_df.iloc[:, 10:-n]
cropped_data = final_df
cropped_data_ = pd.concat([cropped_data,df_tmp], axis=1)
cropped_data_.to_csv('../data/processed/'+ date + '/cropped_data.csv', index = False)


#group by glucose level
cropped_data['glucose_level'] = df_tmp['glucose_level']
cropped_data_300  = cropped_data[ cropped_data['glucose_level'] == 300  ]
cropped_data_1100 = cropped_data[ cropped_data['glucose_level'] == 1100 ]
cropped_data_5000 = cropped_data[ cropped_data['glucose_level'] == 5000 ]

cropped_data_300.drop('glucose_level',inplace = True, axis = 1)
cropped_data_1100.drop('glucose_level',inplace = True, axis = 1)
cropped_data_5000.drop('glucose_level',inplace = True, axis = 1)

fig, ax = plt.subplots()
ax.plot(cropped_data_300.T, color = 'red')
ax.plot(cropped_data_1100.T, color = 'blue')
ax.plot(cropped_data_5000.T, color = 'green')

#add vertical band lines
for i in range(0,len(peaks)):
    x_1 = df_range.iloc[1,i]
    x_2 = df_range.iloc[2,i]
    ax.axvspan(x_1, x_2, alpha=0.5, color='red')




red_patch = mpatches.Patch(color='red',label = 'glucose_level 300')
blue_patch = mpatches.Patch(color='blue',label = 'glucose_level 1100')
green_patch = mpatches.Patch(color='green',label = 'glucose_level 5000')
plt.legend(handles=[red_patch,blue_patch,green_patch])

plt.title('Glucose concentration')
plt.xlabel('wavelength')
plt.ylabel('transmittance')
plt.show()


#plt.plot(cropped_data.T)
#plt.show()
#A = 1





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