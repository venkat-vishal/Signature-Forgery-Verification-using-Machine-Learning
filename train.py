import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from skimage import feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

'''
path = "Dataset"
X = []
Y = []
hog = cv2.HOGDescriptor()

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_features, img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, transform_sqrt=True, block_norm='L1')
            X.append(hog_features)
            label = 0
            if name == 'Forge':
                label = 1
            Y.append(label)
            print(name+" "+str(label)+" "+str(hog_features.shape))
            

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)
print(Y.shape)
print(np.unique(Y, return_counts=True))

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
print(cnn_model.summary())
cnnmodel = Model(cnn_model.inputs, cnn_model.layers[-2].output)#create mobilenet  model
cnn_features = cnnmodel.predict(X)  #extracting mobilenet features
Y1 = np.argmax(Y, axis=1)
print(cnn_features.shape)

dt = DecisionTreeClassifier()
dt.fit(cnn_features, Y1)
importances = dt.feature_importances_

X_train, X_test, y_train, y_test = train_test_split(cnn_features, Y1, test_size=0.2) #split dataset into train and test

rf = svm.SVC()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

rf = KNeighborsClassifier(n_neighbors=2)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)
