import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Input


def LoadCitraTraining(sDir,LabelKelas):
  JumlahKelas=len(LabelKelas)
  TargetKelas = np.eye(JumlahKelas)
  # Menyiapkan variabel list untuk data menampung citra dan data target
  X=[]#Menampung Data Citra
  T=[]#Menampung Target
  for i in range(len(LabelKelas)):
    #Membaca file citra di setiap direktori data set
    DirKelas = os.path.join(sDir, LabelKelas[i])
    files = os.listdir(DirKelas)
    for f in files:
      ff=f.lower()
      print(f)
      #memilih citra dengan extensi jpg,jpeg,dan png
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
         NmFile = os.path.join(DirKelas,f)
         img= np.double(cv2.imread(NmFile,1))
         img=cv2.resize(img,(128,128));
         img= np.asarray(img)/255;
         img=img.astype('float32')
         #Menambahkan citra dan target ke daftar
         X.append(img)
         T.append(TargetKelas[i])
     #--------akhir loop :Pfor f in files-----------------
  #-----akhir  loop :for i in range(len(LabelKelas))----

  #Mengubah List Menjadi numppy array
  X=np.array(X)
  T=np.array(T)
  X=X.astype('float32')
  T=T.astype('float32')
  return X,T

def ModelDeepLearningCNN(JumlahKelas):

    model = Sequential()
    model.add(Input((128,128,3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu',padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu',padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu',padding="same"))

    model.add(Flatten())
    model.add(Dense(JumlahKelas, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #ModelCNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas,NamaFileBobot):
    #Membaca Data training dan label Kelas
    X,D=LoadCitraTraining(DirektoriDataSet,LabelKelas)
    JumlahKelas = len(LabelKelas)
    #Membuat Model CNN
    ModelCNN =ModelDeepLearningCNN(JumlahKelas)
    #Trainng
    history=ModelCNN.fit(X, D,epochs=JumlahEpoh,shuffle=True)
    #Menyimpan hasil learning
    ModelCNN.save(NamaFileBobot)
    #Mengembalikan output
    return ModelCNN,history


def Klasifikasi(DirDataSet,DirKlasifikasi,LabelKelas,ModelCNN=[]):
#Menyiapkan Data input Yang akan di kasifikasikan
  X=[]
  ls = []
  DirKelas = DirDataSet+"/"+DirKlasifikasi
  print(DirKelas)
  files = os.listdir(DirKelas)
  n=0
  for f in files:
      ff=f.lower()
      print(f)
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
         ls.append(ff)
         NmFile = os.path.join(DirKelas,f)
         img= cv2.imread(NmFile,1)
         img=cv2.resize(img,(128,128))
         img= np.asarray(img)/255
         img=img.astype('float32')
         X.append(img)
     

  X=np.array(X)
  X=X.astype('float32')
  #Melakukan prediksi Klasifikasi
  hs=ModelCNN.predict(X)

  LKlasifikasi=[]
  LKelasCitra =[]
  n = X.shape[0]
  for i in range(n):
      v=hs[i,:]
      if v.max()>0.5:
          idx = np.max(np.where( v == v.max()))
          LKelasCitra.append(LabelKelas[idx])
      else:
          idx=-1
          LKelasCitra.append("-")
      #------akhir if
      LKlasifikasi.append(idx)
  #----akhir for
  LKlasifikasi = np.array(LKlasifikasi)
  return ls, hs, LKelasCitra


#Menentukan Direktori Yang menyimpan Data set
DirektoriDataSet="dataset"

#Label Data Set
LabelKelas = []

# Tentukan ranks dan suits untuk setiap putaran
ranks = [str(i) for i in range(2, 11)] + ["jack", "queen", "king", "ace"]
suits = ["diamonds", "hearts", "spades", "clubs"]

# Loop untuk menghasilkan label
for rank in ranks:
    for suit in suits:
        label = f"{rank}_{suit}"
        LabelKelas.append(label)

# Cetak hasil label
print(LabelKelas)

#c. Inisialisasi parameter Training
JumlahEpoh = 12
FileBobot = "DeteksiKartu.h5"
#d. training
ModelCNN,history = TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas ,FileBobot)
ModelCNN.summary()