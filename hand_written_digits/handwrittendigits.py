import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
import keras
# model selection
from sklearn.model_selection import train_test_split


# dl libraraies
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Dense, Activation





# function for testing that digits come out normally
def testOutput(images, labels, title):
    #output 10 random images
    fig, ax = plt.subplots(5, 2)
    fig.set_size_inches(10, 10)
    for i in range(5):
        for j in range(2):
            l = rn.randint(0, len(trainlabels))
            ax[i, j].imshow(images[l], cmap=plt.cm.binary)
            ax[i, j].set_title('Digit: ' + str(labels[l]))
    plt.tight_layout()
    plt.show()

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
testimagearray = np.array(idx2numpy.convert_from_file(testimages))

# train labels
testlabels = 'image_files/t10k-labels.idx1-ubyte'
testlabelarray = idx2numpy.convert_from_file(testlabels)


#create a numpy array for validation set
validationlabels = []
validationimages = []

# split the training data set into training and validation sets
x_train, x_validation, y_train, y_validation =  train_test_split(trainimagearray, trainlabelarray, test_size=0.25, random_state=42)


# test to make sure all 3 arrays work
testOutput(x_train, y_train, 'Training Set')
testOutput(x_validation, y_validation, 'Validation Set')
testOutput(testimagearray, testlabelarray, 'Testing Set')


# output splits
splits = [len(y_train), len(y_validation), len(testlabelarray)]
labels=['Training Set', 'Validation Set', 'Testing Set']
plt.figure(figsize=[12, 6])
y_pos = np.arange(len(labels))
plt.bar(y_pos, splits, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.title('Number of digits per directory')
plt.show()


# resize the array to be rank 4
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
testimagearray = testimagearray.reshape(testimagearray.shape[0], 28, 28, 1)

# reshape labels
test_labels = to_categorical(testlabelarray)
train_labels = to_categorical(y_train)
validation_labels = to_categorical(y_validation)

#start building the convnet model with 3 layers
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.add(Dense(10, activation="softmax"))

# define number of epoch and batch size
batch_size=128
epochs=50


# include data augmnetation to avoid overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images



# define generators
test_generator = datagen.flow(testimagearray, test_labels, batch_size=128)
train_generator = datagen.flow(x_train, train_labels, batch_size=128)
validation_generator = datagen.flow(x_validation, validation_labels, batch_size=128)

#compile the model and output the summary
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# Training the model for 50 epochs
History = model.fit_generator(
    generator=train_generator,
    epochs=50,
    validation_data=validation_generator,
    verbose=1)


# save the model
model.save('handwrittenn-digits.h5')

acc = History.history['accuracy']
val_acc = History.history['val_accuracy']
loss = History.history['loss']
val_loss = History.history['val_loss']

model = load_model('handwritten-digits.h5')

#evaluate the model
test_eval = model.evaluate(testimagearray, test_labels, verbose=1)

# print the loss and accuracy 
print("Loss=", test_eval[0])
print("Accuracy=", test_eval[1])

print("acc \t val_acc \t loss \t val_loss")
print("------------------------------------------------------------------")

for i in range(len(acc)):
    print('{:.3}'.format(acc[i]), " \t", '{:.3}'.format(val_acc[i]), "\t\t",
          '{:.3}'.format(loss[i]), "\t", '{:.3}'.format(val_loss[i]))




# save the history object
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(History.history)

# # save accuracy and loss to a csv if we need to go back ever
hist_csv_file = 'accuracy-vs-loss.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


#output the models performance - loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#output the models -performance - accuracy
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()



y_pred = model.predict(testimagearray)
y_pred = np.argmax(y_pred, axis=1).astype(int)

print(y_pred)

# output some predictions
plt.figure(figsize=(15,15))

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(np.squeeze(testimagearray[i]), cmap=plt.cm.binary)
    actual_label = testlabelarray[i]
    predict_label = y_pred[i]
    plt.title("Actual: " + str(actual_label) + "| Prediction: " + str(predict_label))
plt.show()


#output the stats of correctly/non-correctly identified flowers
corr = []
incorr = []
corr_count = 0
incorr_count = 0

for i in range(len(testlabelarray)):
    if (y_pred[i] == y_true[i]):
        corr.append(i)
        corr_count += 1
    else:
        incorr.append(i)
        incorr_count += 1

print("Found %d correct digits" % (corr_count))
print("Found %d incorrect digits" % (incorr_count))