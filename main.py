import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from lstm_autoencoder import LSTM_Autoencoder
from models import LSTMAutoEncoder

#Beer Production in Australia
df = pd.read_csv("beer_production_australia.csv")
print(df.head(5))

var_name = "production"
# Visualise data
plt.title("Dataset:")
plt.plot(df[var_name])
plt.show()

# Use green area as training data (non-anomalous)
start, end = 50, 100
plt.title("Non-anomalous data (green segment) used as training data")
plt.plot(df[var_name])
plt.plot(df[var_name][start:end], c='g')
plt.show()


# Create a trajectory matrix, i.e. rolling window representation
timeseries_train = df[var_name][start:end]
traj_mat_train = utils.get_window(timeseries_train, backward=4)

timeseries_test = df[var_name]
traj_mat_test = utils.get_window(timeseries_test, backward=4)


lae = LSTMAutoEncoder()
lae.fit(traj_mat_train)
scores = lae.predict(traj_mat_test)

ratio = 0.99
sorted_scores = sorted(scores)
threshold = sorted_scores[round(len(scores) * ratio)]

plt.plot(scores)
plt.plot([threshold]*len(scores), c='r')
plt.title("Reconstruction errors")
plt.show()

anomaly_idx = np.where(scores > threshold)
normal_idx = np.where(scores <= threshold)

plt.scatter(normal_idx, traj_mat_test[normal_idx][:, -1], s=1)
plt.scatter(anomaly_idx, traj_mat_test[anomaly_idx][:, -1], c='r', s=5)
plt.show()
