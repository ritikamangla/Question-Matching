# Semantic Question Matching
Semantic Question Matching with Deep Learning Keras - achieving 0.81 `F1 score`, and 0.86% `Accuracy` on `validation set`.

This is Keras implementation of [Semantic Question Matching with Deep Learning](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)

There was also [Kaggle competition](https://www.kaggle.com/c/quora-question-pairs/data).

## Dataset

Test data: https://www.kaggle.com/c/quora-question-pairs/download/test.csv 

## Word embedding
300 dimensional [Fasttext word embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html) -> wiki.en.vec are used.


# Running code

Create a folder called word_embeddings and add wiki.en.vec in it.

python run.py to run the script.
```


