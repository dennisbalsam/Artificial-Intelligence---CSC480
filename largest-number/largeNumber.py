import sys
import math
import random

# program to represent the law of large numbers, which uses an claims that as the number of trials increases, 
# the actual probability of the trial approaches the expected probability
# i utilize a common trial of flipping a coin many times which should reach a probability of 0.5 or 50%. I will use different number of attempts for this


# some variables
headspercentage = 0.0
tailspercentage = 0.0
value = 1

for iteration in range(0,7):
    value*=10
    # reset values
    heads = 0
    tails = 0
    for i in range(0,value):
        #get random number emulating a coin flip
        flip= random.randint(0,1)
        if flip == 0:
            heads +=1
        else:
            tails+=1
    headspercentage = (heads / value) * 100
    tailspercentage = (tails / value) * 100
    print("Iterations: ", value, " | Heads: ",heads,"|  Tails: ",tails," |  Heads-Percentage=",round(headspercentage, 3), " | Tails-Percentage=",round(tailspercentage, 3)) 

