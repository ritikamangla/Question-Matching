import keras
from keras.models import load_model
import tensorflow as tf
from keras import backend as K

import sqmutils.data_utils as du

import os
import time

import pandas as pd
import numpy as np

import csv

import json

model_dir = "models"
dataset_dir = "dataset"
word_embed = "word_embeddings"
model_weights = os.path.join(model_dir, "best_val_f1_model.h5")
# you can download test data from here:
# https://www.kaggle.com/c/quora-question-pairs/download/test.csv
test_dataset_path = os.path.join(model_dir, "test.csv")
cleaned_test_dataset_path = os.path.join(dataset_dir, "cleaned_test.csv")
test_probabilities_csv = os.path.join(dataset_dir, "test_probabilities.csv")

emb_dim = 300



config = du.get_config(None, None, None,  embedding_dimension=emb_dim)
custom_objects= {"f1": du.f1, "recall" : du.recall, "precision" : du.precision}

start = time.time()
dfTest = pd.read_csv(test_dataset_path, sep=',', encoding='utf-8')
end = time.time()
print("Total time passed", (end - start))
print("Total test examples", len(dfTest))

start = time.time()
valid_ids =[type(x)==int for x in dfTest.test_id]
dfTest = dfTest[valid_ids].drop_duplicates()
dfTest = dfTest.replace(np.nan, '', regex=True)
dfTest = dfTest.fillna('')
dfTest.to_csv(cleaned_test_dataset_path, sep=',', encoding='utf-8', index=False)
end = time.time()
print("Total time passed", (end - start))
print("Total test examples", len(dfTest))

#print(dfTest[:10])

embedding_path = "C://Users//hp//PycharmProjects//semantic-question-matching-master//word_embeddings//wiki.en.vec"

print("word vectors path", embedding_path)
start = time.time()
w2v = du.load_embedding(embedding_path)
end = time.time()
print("Total time passed: ", (end-start))

model = load_model(model_weights, custom_objects = custom_objects)

print("HI")
