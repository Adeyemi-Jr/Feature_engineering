
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

date = '20220211'
fig, (ax, ax_)  = plt.subplots(1, 2, figsize=(12, 6))
#uniform_data = np.random.rand(10, 1)
uniform_data = pd.read_csv('../results/accuracies/accuracy_band_1.csv')
cmap = mpl.cm.Blues_r
ax = sns.heatmap(uniform_data, annot=True, cmap=cmap)


x_count = 0
y_count = 0
# iterate through both the labels and the texts in the heatmap (ax.texts)
for lab, annot in zip(ax.get_yticklabels(), ax.texts):

    text =  lab.get_text()
    print(text)
    current_row = str(uniform_data.index.values[y_count])
    if text == current_row:# lets highlight row 2
        x_count = 0
        for lab_, annot_ in zip(ax.get_xticklabels(), ax.texts):
            text_ = lab_.get_text()
            current_col = str(uniform_data.columns.values[x_count])
            if text_ == current_col:  # lets highlight row 2
                # set the properties of the ticklabel
                #lab.set_weight('bold')
                #lab.set_size(20)
                #lab.set_color('purple')
                # set the properties of the heatmap annot
                annot_.set_weight('bold')
                annot_.set_color('purple')
                annot_.set_size(20)

                plt.show()

            x_count += 1
    y_count += 1


