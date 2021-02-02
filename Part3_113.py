from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
from glob import glob

 
def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    #print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.zeros(shape),
                              data))
        else:
            return np.hstack((np.zeros(shape),
                              data))
    else:
        return data
 
 
 
def awgn(signal, snr):
    N = len(signal)
    noise = np.random.normal(0,1,N)
    signal_power = np.sum(np.abs(signal)*np.abs(signal))/N
    noise_power = np.sum(np.abs(noise)*np.abs(noise))/N
    scale_factor = np.sqrt(signal_power/(10**(snr/10))) #fill this in with the correct expression
    noise = noise * scale_factor
    return noise + signal

audio_files_path = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]
 
 
# max_length of audiofile
T = 3.5
 
index = [i for i in range(len(audio_files_path))]
columns = ['data', 'label']
df_train2 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    out_data = pad_audio(data, fs, T)
    label = os.path.dirname(file_path).split("/")[-1]
    df_train2.loc[i] = [out_data, label]
assert(len(audio_files_path)==398)
y = df_train2.iloc[:, 1].values
X = df_train2.iloc[:, :-1].values
X = np.squeeze(X)
X = np.stack(X, axis=0)
from sklearn import preprocessing
 
labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)
 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.1)
# model building
 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten
from keras.optimizers import Adam
from sklearn import metrics 

 
# set input dimensions to length of input data you will be feeding into the neural network
input_dim = 154350
 
model = Sequential()
model.add(Dense(512, input_dim=input_dim, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=10)
 
_, accuracy = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy on the validation dataset is :", accuracy)
PATH= './Problem_3_data/test_data_3'
# max_length of audiofile
T = 3.5
index = [i for i in range(10)]
columns = ['data']
df_test_3 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
    fs, data = wavfile.read(file_path)
    out_data = pad_audio(data, fs, T)
    df_test_3.loc[i] = [out_data]
X_test = df_test_3.iloc[:, :].values
X_test = np.squeeze(X_test)
X_test = np.stack(X_test, axis=0)
 
##Part a Predicting without DFT Features
########## Your code  ############
##Prediction without DFT Features
y_test = model.predict(X_test)
print(y_test)
 
# set input dimensions to length of input data you will be feeding into the neural network
input_dim = 154350
 
model_dft = Sequential()
model_dft.add(Dense(512, input_dim=input_dim, activation='relu'))
model_dft.add(Dense(256, activation='relu'))
model_dft.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model_dft.add(Dense(1, activation='sigmoid'))
model_dft.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# set input dimensions to length of input data you will be feeding into the neural network
input_dim = 154350
 
model_dft = Sequential()
model_dft.add(Dense(512, input_dim=input_dim, activation='relu'))
model_dft.add(Dense(256, activation='relu'))
model_dft.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model_dft.add(Dense(1, activation='sigmoid'))
model_dft.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
## takes a time series matrix and returns a matrix with dft features from ##problem 1
def dftofsignalMatrix(arr):
  arrdft = np.zeros([arr.shape[0],arr.shape[1]])
  for i in range(arr.shape[0]):
    arr_chord = arr[i]
    N = arr_chord.shape[0] ##Signal Points
    arr_dft = np.fft.fft(arr_chord,N)
    arr_dft_shifted = np.fft.fftshift(arr_dft)
    arrdft[i,:] = np.abs(arr_dft)
  return arrdft
 
##Preprocess DFT X_train,X_val,X_test
xtraindft = dftofsignalMatrix(X_train)
xvaldft = dftofsignalMatrix(X_val)
xtestdft = dftofsignalMatrix(X_test)
 
##Train on DFT
model_dft.fit(xtraindft, y_train, epochs=5, batch_size=10)
 
_, accuracy = model_dft.evaluate(xvaldft, y_val, verbose=0)
print("Accuracy on the validation dataset is :", accuracy)
y_test = model_dft.predict(xtestdft)
print(y_test)
######## your code here #########
import matplotlib.pyplot as plt
 
##Creates Empty Matrices 
noise_5 = np.zeros([X_test.shape[0],X_test.shape[1]])
noise_0 = np.zeros([X_test.shape[0],X_test.shape[1]])
noise_minus5 = np.zeros([X_test.shape[0],X_test.shape[1]])
 
## Adding AWGN to the Test Set
for i in range(X_test.shape[0]):
  noise_5[i,:] = awgn(X_test[i],5)
  noise_0[i,:] = awgn(X_test[i],0)
  noise_minus5[i,:] = awgn(X_test[i],-5)
 
##Taking DFT Features
noise_5dft = dftofsignalMatrix(noise_5)
noise_0dft = dftofsignalMatrix(noise_0)
noise_minus5dft = dftofsignalMatrix(noise_minus5)
 
## Finding Accuracy of Noised Sets
_, accuracy1 = model_dft.evaluate(noise_5dft, y_test, verbose=0)
print("Accuracy on the noise_5dft dataset is :", accuracy1)
_, accuracy2 = model_dft.evaluate(noise_0dft, y_test, verbose=0)
print("Accuracy on the noise_0dft dataset is :", accuracy2)
_, accuracy3 = model_dft.evaluate(noise_minus5dft, y_test, verbose=0)
print("Accuracy on the noise_minus5dft dataset is :", accuracy3)
from scipy import signal
import scipy
##Function that denoises
def denoiser(matrix):
  y = scipy.signal.savgol_filter(matrix, 51, 3,axis = 1) # window size 51, polynomial order 3
  return y
##Denoises Noisy Time Signals
denoised_noise_5 = denoiser(noise_5)
denoised_noise_0 = denoiser(noise_0)
denoised_noise_minus5 = denoiser(noise_minus5)
##DFT of Denoised Time Signals
denoised_noise_5dft = dftofsignalMatrix(denoised_noise_5)
denoised_noise_0dft = dftofsignalMatrix(denoised_noise_0)
denoised_noise_minus5dft = dftofsignalMatrix(denoised_noise_minus5)
##Accraucy of DFT DENOISED
_, accuracy4 = model_dft.evaluate(denoised_noise_5dft, y_test, verbose=0)
_, accuracy5 = model_dft.evaluate(denoised_noise_0dft, y_test, verbose=0)
_, accuracy6 = model_dft.evaluate(denoised_noise_minus5dft, y_test, verbose=0)
print("Accuracy on the denoised_noise_5dft dataset is :", accuracy4)
print("Accuracy on the denoised_noise_0dft dataset is :", accuracy5)
print("Accuracy on the denoised_noise_minus5dft dataset is :", accuracy6)
 
import sys
######## your code here #########
##Clipping The Audio
PATH = './Problem_3_data/training_data_3'
 
audio_files_path = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]
 
 
# max_length of audiofile
 
index = [i for i in range(len(audio_files_path))]
columns = ['data', 'label']
df_train2 = pd.DataFrame(index=index, columns=columns)
minSize = sys.maxsize
for j,file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    if(data.shape[0]<minSize):
      minSize = data.shape[0]
for i, file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    out_data = data[0:minSize]
    label = os.path.dirname(file_path).split("/")[-1]
    df_train2.loc[i] = [out_data, label]
 
y = df_train2.iloc[:, 1].values
X_crop = df_train2.iloc[:, :-1].values
X_crop = np.squeeze(X_crop)
X_crop = np.stack(X_crop, axis=0)
 
from sklearn import preprocessing
 
labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)
 
 
from sklearn.model_selection import train_test_split
X_train_crop, X_val_crop, y_train, y_val = train_test_split(X_crop, y, shuffle=True, test_size=0.1)
 
 
# set input dimensions to length of input data you will be feeding into the neural network
input_dim = minSize
 
model_crop = Sequential()
model_crop.add(Dense(512, input_dim=input_dim, activation='relu'))
model_crop.add(Dense(256, activation='relu'))
model_crop.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model_crop.add(Dense(1, activation='sigmoid'))
 
model_crop.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
PATH= './Problem_3_data/test_data_3'
index = [i for i in range(10)]
columns = ['data']
df_test_3 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
    fs, data = wavfile.read(file_path)
    out_data = data[0:minSize]
    df_test_3.loc[i] = [out_data]
 
X_test_crop = df_test_3.iloc[:, :].values
X_test_crop = np.squeeze(X_test_crop)
X_test_crop = np.stack(X_test_crop, axis=0)
 
 
##Preprocess DFT X_train,X_val,X_test
xtraindftcrop = dftofsignalMatrix(X_train_crop)
xvaldftcrop = dftofsignalMatrix(X_val_crop)
xtestdftcrop = dftofsignalMatrix(X_test_crop)
##Fit Model
model_crop.fit(xtraindftcrop, y_train, epochs=5, batch_size=10)
##validation Accuracy
_, accuracy = model_crop.evaluate(xvaldftcrop, y_val, verbose=0)
print("Accuracy on the validation dataset is :", accuracy)
##test Accuracy
_, accuracy3 = model_crop.evaluate(xtestdftcrop, y_test, verbose=0)
print("Accuracy on the test dataset is :", accuracy3)

