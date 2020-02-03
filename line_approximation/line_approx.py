import sys
import math
import random
import numpy


m=s=0

L=10


for j in range(2,10):
    L=L*10
    s=0
    for i in range(1,L):
        r=random.uniform(-.1,.1)
        s=s+r
        m=1.0*s/(L*1.0)

print("j=",j,"s=",s,"L=",L,"m=",m)