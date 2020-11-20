import sys

import os
import logging
import papermill as pm
import tensorflow as tf

import dataset
from utils import prepare_hparams
from dkn_model import DKN
from dkn_iterator import DKNTextIterator


if __name__ == "__main__":
    print(f"System version: {sys.version}")
    print(f"Tensorflow version: {tf.__version__}")
    
    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%I:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    # prepare data
    behavior_path = "..\\dataset\\MINDsmall_train\\behaviors.tsv"
    train_file_path = "..\\dataset\\data_path\\train_mind.txt"
    valid_file_path = "..\\dataset\\data_path\\valid_mind.txt"
    user_history_path = "..\\dataset\\data_path\\user_history.txt"
    dataset.get_data(behavior_path, train_file_path, valid_file_path, user_history_path)
    train_news = "..\\dataset\\MINDsmall_train\\news.tsv"
    valid_news = "..\\dataset\\MINDsmall_train\\news.tsv"
    news_words, news_entities = dataset.get_words_and_entities(train_news, valid_news)
    data_path = "..\\dataset\\data_path"
    glove_path = "..\\dataset\\glove.6B"
    train_entities = "..\\dataset\\MINDsmall_train\\entity_embedding.vec"
    valid_entities = "..\\dataset\\MINDsmall_train\\entity_embedding.vec"
    dataset.generate_embeddings(glove_path, data_path, news_words, news_entities, train_entities, valid_entities)

    # train
    epochs = 10
    history_size = 50
    batch_size = 100
    yaml_file = "dkn_MINDsmall.yaml"
    news_feature_file = "../dataset/data_path/doc_feature.txt"
    user_history_file = "../dataset/data_path/user_history.txt"
    word_embeddings_file = "../dataset/data_path/word_embeddings_5w_100.npy"
    entity_embeddings_file = "../dataset/data_path/entity_embeddings_5w_100.npy"

    hparams = prepare_hparams(yaml_file,
                     news_feature_file=news_feature_file,
                     user_history_file=user_history_file,
                     wordEmb_file=word_embeddings_file,
                     entityEmb_file=entity_embeddings_file,
                     epochs=epochs,
                     history_size=history_size,
                     batch_size=batch_size)
    
    model = DKN(hparams, DKNTextIterator)
    train_file = "../dataset/data_path/train_mind.txt"
    valid_file = "../dataset/data_path/valid_mind.txt"
    model.fit(train_file, valid_file)

    res = model.run_eval(valid_file)
    print(res)
    pm.record("res", res)
