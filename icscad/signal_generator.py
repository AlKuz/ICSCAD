import numpy as np
import pandas as pd


def random_step(time, freq_steps=1.0, sample_time=0.1, min_val=0.0, max_val=1.0, size=1, seed=1, names=None):
    """
    Function for generating random step signals.
    Initially it has one parameter. If size > 1, then
    freq_steps, sample_time, min_val and max_val are lines.
    :param time: time of signal
    :param freq_steps: frequency of step signals in Hz
    :param sample_time: step time of each value in the array
    :param min_val: minimum value(s). If size > 1 then it is numpy array
    :param max_val: maximum value(s). If size > 1 then it is numpy array
    :param size: number of rows different parameters
    :param seed: seed of random numbers
    :param names: line of parameter names
    :return: pandas data frame
    """
    t = 0.0
    num_in_steps = 1 / freq_steps / sample_time
    n = 0
    if names is None:
        columns = ['t'] + [str(i+1) for i in range(size)]
    else:
        columns = ['t'] + names
    np.random.seed(seed)
    data = np.empty([0, size+1])
    data_line = np.empty([0, size+1])
    while t < time:
        if n % num_in_steps == 0:
            data_line = np.random.uniform(0, 1, size)
            n = 0
        data = np.vstack([data, np.append(t, data_line)])
        n += 1
        t += sample_time
    data = data * np.append([1], np.array(max_val)-np.array(min_val)) + np.append([0], np.array(min_val))
    return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    #a = random_step(50, 5.0, 0.1, [300.0, 10000.0], [500.0, 100000.0], 2, 1, names=['T', 'n'])
    a = random_step(50, sample_time=0.01)
    print(a)
