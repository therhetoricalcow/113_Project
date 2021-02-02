from google.colab import drive
drive.mount('/content/drive')
# importing relevant libraries
from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
from glob import glob
import scipy.signal
# update this with the path of your training_data folder
 
PATH= './Problem_1_data/training_data_1'
audio_file_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]
# since we have a total of 1994 chords i.e. ~200 files belonging to 10 classes
assert(len(audio_file_paths) == 1994)
len(audio_file_paths)
index = [i for i in range(len(audio_file_paths))]
columns = ['data', 'label']
df_train = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(audio_file_paths):
   fs, data = wavfile.read(file_path)
   # label assigned to each chord is the name of the folder it is placed inside
   label = os.path.dirname(file_path).split("/")[-1]
   df_train.loc[i] = [data, label]
y = df_train.iloc[:, 1].values
X = df_train.iloc[:, :-1].values
X = np.squeeze(X)
#X = np.stack(X, axis=0)
from sklearn import preprocessing
 
labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)
######## your pt. a code here #########
import matplotlib.pyplot as plt
X_chosen = []
for i in range(max(y)+1):
 #extract chord for each file
 a = (np.where(y == i)[0][0])
 chord = X[a]
 X_chosen.append(chord)
 N = chord.shape[0] ##Signal Points
 #Find DFT using fft for each chord
 A = np.fft.fft(chord,N)
 shifted_A = np.fft.fftshift(A)
 freq = np.fft.fftfreq(N, 1.0/fs) #frequency for the x axis
 #define range for plots to gather most useful parts of the data
 range_freq = (freq >= 0) & (freq < 3.2e3)
 #plot frequencies in range vs. absolute value of DFT in range
 plt.figure()
 plt.plot(freq[range_freq],np.abs(A[range_freq]))
 plt.title('DFT ' + str(labelencoder_y.classes_.item(i)))
 
######## your pt. b code here #########
###### create your function #######
def convolution_(known, unknown):
 #convolving the known signal with the time reversed unknown signal
 return scipy.signal.fftconvolve(known,unknown[::-1],mode = 'full')
PATH= './Problem_1_data/test_data_1'
# max_length of audiofile
index = [i for i in range(10)]
columns = ['data']
df_test_1 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
   fs, data = wavfile.read(file_path)
   df_test_1.loc[i] = [data]
X_test = df_test_1.iloc[:, :].values
X_test = np.squeeze(X_test)
######## your code here #########
train_0 = 0  #first chord type
train_1 = 1  #second chord type
sig0 = (np.where(y == train_0)[0][0])
sig1 = (np.where(y == train_1)[0][0])
#convolving two chord types into an array
conv = convolution_(X[sig0],X[sig1])
conv
###### Function for Known Sampled Training Data and Training Set #######
indices = []
#define empty confusion matrix
conf_mat = np.zeros([10,10])
for i in range(max(y)+1):
  ind_list = np.where(y==i)[0]
  arr = ind_list[1:].copy()
  indices.append(arr)
 
for i in range(len(indices)):
  for j in range(len(indices[i])):
    vals = np.zeros([1,10])
    for k in range(len(X_chosen)):
      conv = convolution_(X_chosen[k]/np.linalg.norm(X_chosen[k]),X[indices[i][j]]/np.linalg.norm(X[indices[i][j]]))
      N = conv.shape[0]
      val = ((1/N)*np.square(np.linalg.norm(conv)))
      vals[0,k] = val
    right_ind = np.argmax(vals[0])
    conf_mat[right_ind,i] +=1
 
plt.imshow(conf_mat)
plt.title('Confusion Matrix')
plt.xlabel('Labels')
plt.ylabel('Predicted Values')
 
 
######## your pt. c code here #########
###### Function for unknown test Data and Training Set #######
indices = []
#define empty confusion matrix
conf_mat = np.zeros([10,10])
for i in range(max(y)+1):
  ind_list = np.where(y==i)[0]
  arr = ind_list[1:].copy()
  indices.append(arr)
 
for i in range(len(indices)):
  for j in range(len(indices[i])):
    vals = np.zeros([1,10])
    for k in range(len(X_chosen)):
      conv = convolution_(X_chosen[k]/np.linalg.norm(X_chosen[k]),X[indices[i][j]]/np.linalg.norm(X[indices[i][j]]))
      N = conv.shape[0]
      val = ((1/N)*np.square(np.linalg.norm(conv)))
      vals[0,k] = val
    right_ind = np.argmax(vals[0])
    conf_mat[right_ind,i] +=1
 
plt.imshow(conf_mat)
plt.title('Confusion Matrix')
plt.xlabel('Labels')
plt.ylabel('Predicted Values')


