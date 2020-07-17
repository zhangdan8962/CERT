import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch

from sklearn.datasets import load_digits
# digits = load_digits()

# print(digits)

# print(digits.data.shape)

# data_X = digits.data[:600]
#
# print(type(data_X))
# print(data_X.shape)
#
# y = digits.target[:600]
#
# print(type(y))
# print(y.shape)

with open("train_data_f.pkl","rb") as file1:
    train_data = pickle.load(file1)

with open("ttain_label_f.pkl","rb") as file2:
    train_label = pickle.load(file2)

data_X = train_data.detach().numpy()
y = train_label.detach().numpy()

# print(type(train_data_np))
# print(train_data_np.shape)
#
# print(type(train_label_np))
# print(train_label_np.shape)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

tsne_obj= tsne.fit_transform(data_X)

tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'digit':y})
print(tsne_df.head())
palette = sns.color_palette("bright", 80)
sns.scatterplot(x="X", y="Y",
              hue="digit",
              palette=palette,
              legend='full',
              data=tsne_df)

plt.show()