import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random as rn

# extract files from .gz file

# ------------- training -----------------------------
# train images
trainimages = 'image_files/train-images.idx3-ubyte'
trainimagearray = idx2numpy.convert_from_file(trainimages)

# train labels
trainlabels = 'image_files/train-labels.idx1-ubyte'
trainlabelarray = idx2numpy.convert_from_file(trainlabels)

# ------------- testing ------------------------------
# train images
testimages = 'image_files/t10k-images.idx3-ubyte'
testimagearray = idx2numpy.convert_from_file(testimages)

# train labels
testlabels = 'image_files/t10k-labels.idx1-ubyte'
testlabelarray = idx2numpy.convert_from_file(testlabels)


#output 10 random images
fig, ax = plt.subplots(5, 2)
fig.set_size_inches(10, 10)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(trainlabels))
        ax[i, j].imshow(trainimagearray[l], cmap=plt.cm.binary)
        ax[i, j].set_title('Digit: ' + str(trainlabelarray[l]))
plt.tight_layout()
plt.show()
