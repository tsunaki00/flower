from PIL import Image
import sys
from sklearn import cross_validation
import os, glob
import numpy as np
import random, math

#
# トレーニングデータを生成
#
class DataCreate : 
  def __init__(self, script_name):
    Image.LOAD_TRUNCATED_IMAGES = True
  
  def create(self) :
    input_dir = "images"
    categorys = []
    
    dir_list = os.listdir(input_dir)
    for index, dir_name in enumerate(dir_list):
      if dir_name == '.DS_Store' :
        continue
      categorys.append(dir_name)
    image_size = 50
    train_data = [] # 画像データ, ラベルデータ
    for idx, category in enumerate(categorys): 
      try :
        print("---", category)
        image_dir = input_dir + "/" + category
        files = glob.glob(image_dir + "/*.jpg")
        for i, f in enumerate(files):
          img = Image.open(f)
          img = img.convert("RGB")
          img = img.resize((image_size, image_size))
          data = np.asarray(img)
          train_data.append([data, idx])
      except:
        print("SKIP : " + category)

    # データをshuffle
    random.shuffle(train_data)
    X, Y = [],[]
    for data in train_data: 
      X.append(data[0])
      Y.append(data[1])

    test_idx = math.floor(len(X) * 0.8)
    xy = (np.array(X[0:test_idx]), np.array(X[test_idx:]), 
          np.array(Y[0:test_idx]), np.array(Y[test_idx:]))
    np.save("./npy/flower", xy)
  

if __name__ == "__main__":
  args = sys.argv
  datacreate = DataCreate(args[0])
  datacreate.create()


