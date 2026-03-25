from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2
import os
import numpy as np
#loading python require packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPooling3D, LSTM,RepeatVector
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from skimage import feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from keras.callbacks import ModelCheckpoint
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Convolution2D


main = tkinter.Tk()
main.title("Signature Forgery Verification") #designing main screen
main.geometry("1000x650")


global filename, X, Y
global X_train, X_test, y_train, y_test, cnn_model
global accuracy, precision, recall, fscore

def loadDataset():
    global X, Y
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
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
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X.txt.npy",X)
        np.save("model/Y.txt.npy",Y)

def uploadDataset():
    global filename, X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    X = []
    Y = []
    loadDataset()
    text.insert(END,"Image Processing & HOG Features extracted Completed\n\n")
    text.insert(END,"Total images loaded = "+str(X.shape[0])+"\n\n")
    text.insert(END,"Number of HOG features extracted from each image = "+str(X.shape[1]))

def processDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Image Shuffling & Normalization Completed\n\n")
    text.insert(END,"Dataset Training & Testing Details\n\n")
    text.insert(END,"80% images for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    labels = ['Original', 'Forge']
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_tpr, ns_fpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_tpr, ns_fpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()

def runCNN():
    global X_train, X_test, y_train, y_test, cnn_model, X, Y
    text.delete('1.0', END)
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
    X = cnnmodel.predict(X)  #extracting mobilenet features
    Y = np.argmax(Y, axis=1)
    text.insert(END,"CNN & HOG Based Features Extraction Completed\n\n")
    text.insert(END,"Total features extracted from each image is : "+str(X.shape[1])+"\n\n")

def runDT():
    global X, Y
    dt = DecisionTreeClassifier()
    dt.fit(X, Y)
    importances = dt.feature_importances_
    threshold = 0.0005
    selected_features = [i for i, importance in enumerate(importances) if importance > threshold]
    selected_features = np.asarray(selected_features)
    text.insert(END,"Decision Tree Based Features Selection Completed\n")
    text.insert(END,"Total features Selected by Decision Tree = "+str(selected_features.shape[0])+"\n\n")

def runSVM():
    text.delete('1.0', END)
    global X, Y
    global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", y_test, predict)

def runKNN():
    global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
    knn_cls = KNeighborsClassifier(n_neighbors=2)
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN", y_test, predict)

def runLSTM():
    global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
    X_train1 = np.reshape(X_train, (X_train.shape[0], 16, 16))
    X_test1 = np.reshape(X_test, (X_test.shape[0], 16, 16))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train1, y_train1, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test1, batch_size=1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    calculateMetrics("LSTM", y_test1, predict)    
    

def graph():
    global accuracy, precision, recall, fscore, rmse
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                       ['LSTM','Precision',precision[2]],['LSTM','Recall',recall[2]],['LSTM','F1 Score',fscore[2]],['LSTM','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def predict():
    global  cnn_model
    labels = ['Original', 'Forge']
    filename = filedialog.askopenfilename(initialdir = "testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features, img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, transform_sqrt=True, block_norm='L1')
    X = []
    X.append(hog_features)
    X = np.asarray(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    predict = cnn_model.predict(X)
    predict = np.argmax(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 300))
    cv2.putText(img, "Predicted Output : "+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()       
    

font = ('times', 16, 'bold')
title = Label(main, text='Signature Forgery Verification', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Signature Forgery Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Process Dataset", command=processDataset)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

cnnButton = Button(main, text="CNN & HOG Based Features Extraction", command=runCNN)
cnnButton.place(x=580,y=100)
cnnButton.config(font=font1) 

dtButton = Button(main, text="Decision Tree Based Features Selection", command=runDT)
dtButton.place(x=10,y=150)
dtButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=330,y=150)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=580,y=150)
knnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=10,y=200)
lstmButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=330,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Forge Detection from Test Image", command=predict)
predictButton.place(x=580,y=200)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='blue')
main.mainloop()
