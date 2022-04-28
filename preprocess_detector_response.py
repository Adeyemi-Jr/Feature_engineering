import numpy as np
import pandas as pd
from my_functions import find_nearest



#import data
data = pd.read_csv('../data/processed/20220211/Round_1_transmittance.csv')
detector_resp = pd.read_csv('../data/raw/detector_response.csv')

detector_resp = detector_resp.to_numpy()

wavelengths = data.columns[:-4].to_numpy()
wavelengths  = [float(i) for i in wavelengths]


new_detector_resp = []
for i in range(len(wavelengths)):


    wavelength = wavelengths[i]
    indexes =  find_nearest(detector_resp[:,0], wavelength, K=2)
    indexes  = sorted(indexes)
    index_1 = indexes[0]
    index_2 = indexes[1]

    x_points= [detector_resp[index_1, 0], detector_resp[index_2, 0]]

    y_points= [detector_resp[index_1, 1], detector_resp[index_2, 1]]

    new_detector_resp_ = float(np.interp(wavelength, x_points, y_points))


    #if value out of range of wavelengths set to zero
    if wavelength > max(detector_resp[:,0]):
        new_detector_resp.append(0)
    else:
        new_detector_resp.append(new_detector_resp_)

new_detector_resp_df = pd.DataFrame(new_detector_resp)
new_detector_resp_df = new_detector_resp_df.T
new_detector_resp_df.columns = wavelengths
new_detector_resp_df.to_csv('../data/processed/detector_response_interp.csv')

