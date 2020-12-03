import os
import random
import logging
import json
import numpy as np
from nltk.tokenize import RegexpTokenizer

logger = logging.getLogger()


def _newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def get_data(behaviors_path, train_file_path, valid_file_path, user_history_path, npratio=4):
    """Generate train file, validation file and user history file.
    Args:
       behavior_path (str): Filename.
       train_file_path (str): Path to file.
       valid_file_path (str): Path to file.
       user_history_path (str): Path to file.
       npratio (int): Ratio for negative sampling.
   """
    userid_history = {}
    with open(behaviors_path) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split("\t")
        clicks = click.split(" ")
        pos = []
        neg = []
        imps = imps.split(" ")
        for imp in imps:
            if imp.split("-")[1] == "1":
                pos.append(imp.split("-")[0])
            else:
                neg.append(imp.split("-")[0])
        userid_history[userid] = clicks
        sessions.append([userid, clicks, pos, neg])

    fp_train = open(train_file_path, "w", encoding="utf-8")
    for sess_id in range(len(sessions)):
        userid, _, poss, negs = sessions[sess_id]
        for i in range(len(poss)):
            pos = poss[i]
            neg = _newsample(negs, npratio)
            fp_train.write("1 " + "train_" + userid + " " + pos + "\n")
            for neg_ins in neg:
                fp_train.write("0 " + "train_" + userid + " " + neg_ins + "\n")
    fp_train.close()

    fp_valid = open(valid_file_path, "w", encoding="utf-8")
    for sess_id in range(len(sessions)):
        userid, _, poss, negs = sessions[sess_id]
        for i in range(len(poss)):
            fp_valid.write(
                "1 " + "valid_" + userid + " " + poss[i] + "%" + str(sess_id) + "\n"
            )
        for i in range(len(negs)):
            fp_valid.write(
                "0 " + "valid_" + userid + " " + negs[i] + "%" + str(sess_id) + "\n"
            )
    fp_valid.close()

    fp_user_history = open(user_history_path, "w", encoding="utf-8")
    for userid in userid_history:
        fp_user_history.write(
            "train_" + userid + " " + ",".join(userid_history[userid]) + "\n"
        )
    for userid in userid_history:
        fp_user_history.write(
            "valid_" + userid + " " + ",".join(userid_history[userid]) + "\n"
        )
    fp_user_history.close()


"""
news.tsv
The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file. 
It has 7 columns, which are divided by the tab symbol:
News ID
Category
SubCategory
Title
Abstract
URL
Title Entities (entities contained in the title of this news)
Abstract Entities (entities contained in the abstract of this news)
"""


def _read_news(filepath, news_words, news_entities, tokenizer):
    """
    Read Title words ands entities(including 'SurfaceForms' and 'WikidataId')
    (In future, maybe we can use abstract words and abstract entities)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            splitted = line.strip().split("\t")
            news_words[splitted[0]] = tokenizer.tokenize(splitted[3].lower())
            news_entities[splitted[0]] = []
            for entity in json.loads(splitted[6]):
                news_entities[splitted[0]].append(
                    (entity["SurfaceForms"], entity["WikidataId"])
                )
    return news_words, news_entities


def get_words_and_entities(train_news, valid_news):
    """Load words and entities
    Args:
        train_news (str): News train file.
        valid_news (str): News validation file.
    Returns:
        dict, dict: Words and entities dictionaries.
    """
    news_words = {}
    news_entities = {}
    tokenizer = RegexpTokenizer(r"\w+")
    news_words, news_entities = _read_news(
        train_news, news_words, news_entities, tokenizer
    )
    news_words, news_entities = _read_news(
        valid_news, news_words, news_entities, tokenizer
    )
    return news_words, news_entities


"""
The entity_embedding.vec and relation_embedding.vec files contain 
the 100-dimensional embeddings of the entities and relations learned
from the subgraph (from WikiData knowledge graph) by TransE method. 
In both files, the first column is the ID of entity/relation, and 
the other columns are the embedding vector values.
"""


def generate_embeddings(
        glove_path,
        data_path,
        news_words,
        news_entities,
        train_entities,
        valid_entities,
        max_sentence=10,
        word_embedding_dim=100,
):
    """Generate embeddings.
    Args:
        glove_path (str): glove file path.
        data_path (str): Data path.
        news_words (dict): News word dictionary.
        news_entities (dict): News entity dictionary.
        train_entities (str): Train entity file.
        valid_entities (str): Validation entity file.
        max_sentence (int): Max sentence size.
        word_embedding_dim (int): Word embedding dimension.
    Returns:
        str, str, str: File paths to news, word and entity embeddings.
    """
    embedding_dimensions = [50, 100, 200, 300]
    if word_embedding_dim not in embedding_dimensions:
        raise ValueError(
            f"Wrong embedding dimension, available options are {embedding_dimensions}"
        )

    word_set = set()
    word_embedding_dict = {}
    entity_embedding_dict = {}

    logger.info(f"Loading glove with embedding dimension {word_embedding_dim}...")
    glove_file = "glove.6B." + str(word_embedding_dim) + "d.txt"
    fp_pretrain_vec = open(os.path.join(glove_path, glove_file), "r", encoding="utf-8")
    for line in fp_pretrain_vec:
        linesplit = line.split(" ")
        word_set.add(linesplit[0])
        word_embedding_dict[linesplit[0]] = np.asarray(list(map(float, linesplit[1:])))
    fp_pretrain_vec.close()

    logger.info("Reading train entities...")
    fp_entity_vec_train = open(train_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_train:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_train.close()

    logger.info("Reading valid entities...")
    fp_entity_vec_valid = open(valid_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_valid:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_valid.close()

    logger.info("Generating word and entity indexes...")
    word_dict = {}
    word_index = 1
    news_word_string_dict = {}
    news_entity_string_dict = {}
    entity2index = {}
    entity_index = 1
    for doc_id in news_words:
        news_word_string_dict[doc_id] = [0 for n in range(max_sentence)]
        news_entity_string_dict[doc_id] = [0 for n in range(max_sentence)]
        surfaceform_entityids = news_entities[doc_id]
        for item in surfaceform_entityids:
            if item[1] not in entity2index and item[1] in entity_embedding_dict:  # wikidataID
                entity2index[item[1]] = entity_index
                entity_index = entity_index + 1
        for i in range(len(news_words[doc_id])):
            if news_words[doc_id][i] in word_embedding_dict:
                if news_words[doc_id][i] not in word_dict:
                    word_dict[news_words[doc_id][i]] = word_index
                    word_index = word_index + 1
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                else:
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                for item in surfaceform_entityids:
                    for surface in item[0]:
                        for surface_word in surface.split(" "):
                            if news_words[doc_id][i] == surface_word.lower():
                                if item[1] in entity_embedding_dict:
                                    news_entity_string_dict[doc_id][i] = entity2index[
                                        item[1]
                                    ]
            if i == max_sentence - 1:
                break

    logger.info("Generating word embeddings...")
    word_embeddings = np.zeros([word_index, word_embedding_dim])
    for word in word_dict:
        word_embeddings[word_dict[word]] = word_embedding_dict[word]

    logger.info("Generating entity embeddings...")
    entity_embeddings = np.zeros([entity_index, word_embedding_dim])
    for entity in entity2index:
        entity_embeddings[entity2index[entity]] = entity_embedding_dict[entity]

    news_feature_path = os.path.join(data_path, "doc_feature.txt")
    logger.info(f"Saving word and entity features in {news_feature_path}")
    fp_doc_string = open(news_feature_path, "w", encoding="utf-8")
    for doc_id in news_word_string_dict:
        fp_doc_string.write(
            doc_id
            + " "
            + ",".join(list(map(str, news_word_string_dict[doc_id])))
            + " "
            + ",".join(list(map(str, news_entity_string_dict[doc_id])))
            + "\n"
        )

    word_embeddings_path = os.path.join(
        data_path, "word_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {word_embeddings_path}")
    np.save(word_embeddings_path, word_embeddings)

    entity_embeddings_path = os.path.join(
        data_path, "entity_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {entity_embeddings_path}")
    np.save(entity_embeddings_path, entity_embeddings)

    return news_feature_path, word_embeddings_path, entity_embeddings_path


if __name__ == "__main__":
    behavior_path = "..\\dataset\\MINDsmall_train\\behaviors.tsv"
    train_file_path = "..\\dataset\\data_path\\train_mind.txt"
    valid_file_path = "..\\dataset\\data_path\\valid_mind.txt"
    user_history_path = "..\\dataset\\data_path\\user_history.txt"
    get_data(behavior_path, train_file_path, valid_file_path, user_history_path)
    train_news = "..\\dataset\\MINDsmall_train\\news.tsv"
    valid_news = "..\\dataset\\MINDsmall_train\\news.tsv"
    news_words, news_entities = get_words_and_entities(train_news, valid_news)
    data_path = "..\\dataset\\data_path"
    glove_path = "..\\dataset\\glove.6B"
    train_entities = "..\\dataset\\MINDsmall_train\\entity_embedding.vec"
    valid_entities = "..\\dataset\\MINDsmall_train\\entity_embedding.vec"
    generate_embeddings(glove_path, data_path, news_words, news_entities, train_entities, valid_entities)

