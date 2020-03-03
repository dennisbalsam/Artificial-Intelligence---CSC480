import random
import sys


# eight queens problem is the problem of placing eight queens on an 8×8 chessboard such that 
# none of them attack one another 

#functions


# create tree for traversel 
class Tree:
    def init(self):
        self.right = None
        self.left=None
        self.data=None

root= Tree()
root.data = "initiate tree"

print(root.data)


# driver/entry point