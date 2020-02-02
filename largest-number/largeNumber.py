import sys
import math
import random
from tqdm import tqdm

#import numpy


#a=numpy.arange(100).reshape(1)
m=s=0
L=10


for j in tqdm(range(2,10), desc="outerloop"):
    L=L*10
    s=0
    for i in tqdm(range(1,L),desc="innerloop"):
        r=random.uniform(-.1,.1)
        s=s+r
        m=1.0*s/(L*1.0)


print("j=",j,"s=",s,"L=",L,"m=",m)