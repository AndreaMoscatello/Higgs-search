"""
Code by Andrea Moscatello
Github: https://github.com/AndreaMoscatello/Higgs-search

The dataset is provided and described in: arXiv:1402.4735v2
Two different NN models are provided: the model proposed in the above mentioned paper ("paper") and the model I have designed and tuned ("mine").

Created: January 2021
Last update: January 2025
"""


import warnings
warnings.filterwarnings("ignore")                    # just ignore warnings

# data handling
import numpy as np                                   # linear algebra
import pandas as pd                                  # data processing

# ML learning
from sklearn.linear_model import LogisticRegression  # load the logistic regression model
from sklearn.preprocessing import MinMaxScaler       # scale the input features
from sklearn.model_selection import train_test_split # split data into training and test sets
from sklearn.metrics import confusion_matrix         # calculate & plot confusion matrix

#Neural network
import tensorflow as tf

#from keras.utils.vis_utils import plot_model
import keras

# data visualization
import matplotlib.pyplot as plt                     # basic plotting library
import seaborn as sns                               # more advanced visual plotting library

print("libraries imported")

# read the csv file
#number_of_rows = 4*10**6
number_of_rows = 100_000
n_epochs = 15

#I dont' have an header, so I must add it
header = ["label",
          "lepton pT",
          "lepton eta",
          "lepton phi",
          "missing energy magnitude",
          "missing energy phi",
          "jet 1 pt",
          "jet 1 eta",
          "jet 1 phi",
          "jet 1 b-tag",
          "jet 2 pt",
          "jet 2 eta",
          "jet 2 phi",
          "jet 2 b-tag",
          "jet 3 pt",
          "jet 3 eta",
          "jet 3 phi",
          "jet 3 b-tag",
          "jet 4 pt",
          "jet 4 eta",
          "jet 4 phi",
          "jet 4 b-tag",
          "m_jj",
          "m_jjj",
          "m_lv",
          "m_jlv",
          "m_bb",
          "m_wbb",
          "m_wwbb"]

DataFrame = pd.read_csv("dataset/HIGGS.csv", names = header, nrows=number_of_rows)

Y = DataFrame["label"].values
DataFrame.drop(["label"], axis="columns", inplace=True)
print("DF imported")

X = DataFrame.values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)
#Normalization generally leads to faster learning/convergence - a good practice!
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test  = tf.keras.utils.normalize(X_test, axis=1)


# define which model I want
NNModel = "paper" #mine or paper
if NNModel == "mine":
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="BinaryCrossentropy", #this loss works best
                optimizer=opt,
                metrics=["accuracy"])
    history = model.fit(X_train, Y_train, epochs = n_epochs, validation_data = (X_test,Y_test))
elif NNModel == "paper":
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="BinaryCrossentropy", #this loss works best
                optimizer=opt,
                metrics=["accuracy"])
    history = model.fit(X_train, Y_train, epochs = n_epochs, validation_data = (X_test,Y_test))

# evaluate the model
val_loss, val_acc = model.evaluate(X_test, Y_test)
print(val_loss, val_acc)

#plots
print(history.history.keys())
x_values = np.arange(n_epochs)+1

plt.plot(x_values,history.history['accuracy'], label = 'Train accuracy', color = 'r')
plt.plot(x_values,history.history['val_accuracy'], label = 'Test accuracy', color = 'k')
plt.plot(x_values,history.history['loss'], label = 'Train loss', color = 'r', ls = "--")
plt.plot(x_values,history.history['val_loss'], label = 'Test loss', color = 'k', ls = "--")
plt.title('All-level features')
plt.ylabel('Accuracy/loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("results/all_curves.png")
plt.show()

#output to file
f = open("results/results_all.txt", "w")
f.write(f"val_loss: {val_loss}, val_acc: {val_acc}")
f.close()
tf.keras.utils.plot_model(model, show_shapes=True, to_file = "results/model.png")
