import sys
import math
import random
import numpy


for x in range(0,10):
    r=random.uniform(-.1,.1)
    y = x + (1+r)
    print("Iteration: ", x, " | ", "Equation: " , "y=", y)