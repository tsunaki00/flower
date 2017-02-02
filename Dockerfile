FROM continuumio/miniconda3
  RUN apt-get update
  RUN apt-get install -y mecab libmecab-dev mecab-ipadic
  RUN apt-get install -y mecab-ipadic-utf8
  RUN apt-get install -y libc6-dev build-essential
  RUN pip install mecab-python3
  RUN pip install --upgrade pip
  RUN pip install tensorflow \
                  pandas \
                  scipy \
                  Pillow \
                  keras \
                  h5py
  RUN pip install -U scikit-learn
  RUN mkdir /home/python/
