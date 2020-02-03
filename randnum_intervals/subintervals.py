import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style

# Write a problem to compute 10^6 pseudo random numbers −0.1 ≤
# r ≤ 0.1. Divide the interval [−0.1, 0.1] into 100 equal intervals. Count the
# number of pseudo random numbers in each subinterval. A graphical presentation
# would be very instructive.

#   function for outputting the amount of random numbers in each intervals
def plot_total_values(number_classes, sub_labels):
    # bar graph
    plt.figure(figsize=[12, 6])
    plt.bar(y_pos, number_classes, align='center', alpha=0.5)
    plt.xticks(y_pos, sub_labels, rotation='vertical')
    plt.title('Number of psuedo random numbers in each interval')
    plt.show()


# create the intervals
subintervals = np.arange(start=-0.1, stop = 0.1,step=0.0020,dtype=list)
subintervals = np.append(subintervals, 0.1)
intervals = []

#array to obtain just size of each subinterval
num_in_each = []

# loop to create associative array of arrays
for i in range(101):
    intervals.append([])
    subintervals[i] = round(subintervals[i],3)



print(intervals)
for i in range(0,100000):
    # generate random number in range
    r = random.uniform(-.1, .1)
    # find closest number to it in array
    item_index = np.where(subintervals == min(subintervals, key=lambda x:abs(x-r)))
    # add to count
    intervals[item_index[0][0]].append(1)

# for plot
y_pos = np.arange(len(intervals))
#add to array of just sizes
for i in range(101):
    num_in_each.append(len(intervals[i]))

# call plot function
plot_total_values(num_in_each, subintervals)
