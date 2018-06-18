import numpy as np
import matplotlib.pyplot as plt

# Load engine data
data_JetCat = np.genfromtxt('../data/Data_JetCat_P60.csv', delimiter=',')
data_time = data_JetCat[:, 0]
data_fuel = data_JetCat[:, 1]
data_freq = data_JetCat[:, 2]
data_temp = data_JetCat[:, 3]


# Neural network structure
num_inp = 1
num_hid = 10
num_out = 1
num_out_delays = 1


print(data_JetCat[0, :])
print(len(data_JetCat))

plt.plot(data_time, data_temp)
plt.show()
