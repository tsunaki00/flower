import sys
import os
import numpy as np
import pandas as pd
import gc
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

#
# モデルを生成
#
class TrainModel : 
  def __init__(self):
    input_dir = 'images'
    self.nb_classes = len([name for name in os.listdir(input_dir) if name != ".DS_Store"])
    x_train, x_test, y_train, y_test = np.load("./npy/flower.npy")
    # データを正規化する
    self.x_train = x_train.astype("float") / 256
    self.x_test = x_test.astype("float") / 256
    self.y_train = np_utils.to_categorical(y_train, self.nb_classes)
    self.y_test = np_utils.to_categorical(y_test, self.nb_classes)

  def train(self, input=None) :
    model = Sequential()
    if input == None :
      model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=self.x_train.shape[1:]))
    else :
      model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    if input == None :
      # 学習してモデルを保存
      model.fit(self.x_train, self.y_train, batch_size=32, nb_epoch=10)
      hdf5_file = "./model/flower-model.hdf5"
      model.save_weights(hdf5_file)

      # modelのテスト
      score = model.evaluate(self.x_test, self.y_test)
      print('loss=', score[0])
      print('accuracy=', score[1])
    
    return model

if __name__ == "__main__":
  args = sys.argv
  train = TrainModel()
  train.train()
  gc.collect()

