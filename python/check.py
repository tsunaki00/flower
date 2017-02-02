import train as train
import sys, os
from PIL import Image
import numpy as np
import pandas as pd

if len(sys.argv) <= 1:
  quit()

image_size = 50
input_dir = 'images'
categories = [name for name in os.listdir(input_dir) if name != ".DS_Store"]

X = []
for file_name in sys.argv[1:]:
  img = Image.open(file_name)
  img = img.convert("RGB")
  img = img.resize((image_size, image_size))
  in_data = np.asarray(img)
  X.append(in_data)

X = np.array(X)

model = train.TrainModel().train(X.shape[1:])
model.load_weights("./model/flower-model.hdf5")

predict = model.predict(X)

for pre in predict:
  y = pre.argmax()
  print("花の名前 : ", categories[y])




