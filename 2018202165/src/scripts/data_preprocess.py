from config import Config
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from os import path
import re


def clean_dataset(behaviors_source, behaviors_target, news_source,
                  news_target):
    """
    Remove unnecessary information in MIND dataset
    Args:
        behaviors_source: behaviors.tsv file in MIND dataset
            example:
                U1	11/15/2019 7:28:25 AM	N8240 N61893 N17763 N15442 N45269 N27247 N2807 N64305 N58244 N62099 N1211 N7544	N18774-1 N55354-0
        behaviors_target:
            example:
                clicked_news	candidate_news	clicked
                N12142 N55361 N42151 N5313 N38326 N60863 N32104 N36290 N65 N43756 N1686 N54143 N64745 N54637 N56978 N26686 N31733 N31851 N32288 N57578 N39175 N22904 N9874 N7544 N7228 N61247 N39144 N28742 N10369 N12912 N29465 N38587 N49827 N35943	N11611	0

        news_source: news.tsv file in MIND dataset
            example:
                N1	lifestyle	lifestylefamily	College gymnast dies following practice accident in Connecticut		https://www.msn.com/en-us/lifestyle/lifestylefamily/college-gymnast-dies-following-practice-accident-in-connecticut/ar-BBWBre3?ocid=chopendata	[{"Label": "Connecticut", "Type": "G", "WikidataId": "Q779", "Confidence": 0.999, "OccurrenceOffsets": [54], "SurfaceForms": ["Connecticut"]}]
        news_target:
            example:
                id	title	entities
                N1	College gymnast dies following practice accident in Connecticut	"[{""Label"": ""Connecticut"", ""Type"": ""G"", ""WikidataId"": ""Q779"", ""Confidence"": 0.999, ""OccurrenceOffsets"": [54], ""SurfaceForms"": [""Connecticut""]}]"
    """
    print(f"Clean up {behaviors_source}")
    behaviors = pd.read_table(behaviors_source,
                              header=None,
                              usecols=[3, 4],
                              names=['clicked_news', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()
    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    behaviors['candidate_news'], behaviors[
        'clicked'] = behaviors.impressions.str.split('-').str
    behaviors.clicked_news.fillna('', inplace=True)
    behaviors.to_csv(behaviors_target,
                     sep='\t',
                     index=False,
                     columns=['clicked_news', 'candidate_news', 'clicked'])

    print(f"Clean up {news_source}")
    news = pd.read_table(news_source,
                         header=None,
                         usecols=[0, 3, 6],
                         names=['id', 'title', 'entities'])
    news.to_csv(news_target, sep='\t', index=False)


def balance(source, target, true_false_division_range):
    """
    Args:
        source: file path of original behaviors tsv file
        target: file path of balanced behaviors tsv file
        true_false_division_range: (low, high), len(true_part) / len(false_part) will be within the range
    """
    low = true_false_division_range[0]
    high = true_false_division_range[1]
    assert low <= high
    original = pd.read_table(source)
    true_part = original[original['clicked'] == 1]
    false_part = original[original['clicked'] == 0]
    if len(true_part) / len(false_part) < low:
        print(
            f'Drop {len(false_part) - int(len(true_part) / low)} from false part'
        )
        false_part = false_part.sample(n=int(len(true_part) / low))
    elif len(true_part) / len(false_part) > high:
        print(
            f'Drop {len(true_part) - int(len(false_part) * high)} from true part'
        )
        true_part = true_part.sample(n=int(len(false_part) * high))

    balanced = pd.concat([true_part,
                          false_part]).sample(frac=1).reset_index(drop=True)
    balanced.to_csv(target, sep='\t', index=False)


def parse_news(source, target, word2int_path, entity2int_path, mode):
    """
    Args:
        source: path of tsv file as input
            example:
            id: N1
            title: College gymnast dies following practice accident in Connecticut
            entities: "[{""Label"": ""Connecticut"", ""Type"": ""G"", ""WikidataId"": ""Q779"",
                    ""Confidence"": 0.999, ""OccurrenceOffsets"": [54], ""SurfaceForms"": [""Connecticut""]}]"
        target: path of tsv file
            example:
                id	title	entities
                N1	[1, 2, 3, 4, 5, 6, 7, 8, 0, 0]	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if mode == 'train':
            word2int_path: where to load
            entity2int_path: where to load
        elif mode == 'test':
            word2int_path: where to load
            entity2int_path: where to load
    """
    def clean_text(text):
        return re.sub(r'[^a-zA-Z ]', '', text).lower().strip()

    if mode == 'train':
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        news = pd.read_table(source)
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        with tqdm(total=len(news), desc="Counting words and entities") as pbar:
            for row in news.itertuples(index=False):
                for w in clean_text(row.title).split():
                    if w not in word2freq:
                        word2freq[w] = 1
                    else:
                        word2freq[w] += 1
                for e in json.loads(row.entities):
                    # Count occurrence time within title
                    times = len(
                        list(
                            filter(lambda x: x < len(row.title),
                                   e['OccurrenceOffsets']))) * e['Confidence']
                    if times > 0:
                        if e['WikidataId'] not in entity2freq:
                            entity2freq[e['WikidataId']] = times
                        else:
                            entity2freq[e['WikidataId']] += times
                pbar.update(1)

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= Config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        with tqdm(total=len(news), desc="Parsing words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, [0] * Config.num_words_a_news,
                    [0] * Config.num_words_a_news
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                try:
                    for i, w in enumerate(clean_text(row.title).split()):
                        if w in word2int:
                            new_row[1][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)
        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_word_tokens` in `src/config.py` into 1 + {len(word2int)}'
        )
        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
    elif mode == 'test':
        news = pd.read_table(source)
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        word2int = dict(pd.read_table(word2int_path).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news), desc="Parsing words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, [0] * Config.num_words_a_news,
                    [0] * Config.num_words_a_news
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                try:
                    for i, w in enumerate(clean_text(row.title).split()):
                        word_total += 1
                        if w in word2int:
                            new_row[1][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)
        print(f'Out-of-Vocabulary rate: {word_missed/word_total:.4f}')
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def transform_entity_embedding(source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
            example:
                Q100	-0.075855	-0.164252	0.128812	-0.022738	-0.127613	...
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector']].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int, on='entity').sort_values('int')
    # some entity in entity2int cannot be found in entity_embedding
    entity_embedding_transformed = np.zeros((len(entity2int) + 1, Config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)


def transform_word_embedding(source, target, word2int_path):
    """
    Args:
        source: path of embedding file
            example:
                the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141   ...
        target: path of transformed embedding file in numpy format
        word2int_path
    """
    word_embedding = pd.read_table(source, header=None, sep=' ', engine='python', error_bad_lines=False)
    print(word_embedding.shape)
    word_embedding['vector'] = word_embedding.iloc[:, 1:101].values.tolist()
    word_embedding = word_embedding[[0, 'vector']].rename(columns={0: "word"})

    word2int = pd.read_table(word2int_path)
    merged_df = pd.merge(word_embedding, word2int, on='word').sort_values('int')
    word_embedding_transformed = np.zeros((len(word2int) + 1, Config.word_embedding_dim))
    for row in merged_df.itertuples(index=False):
        word_embedding_transformed[row.int] = row.vector
    np.save(target, word_embedding_transformed)


def transform2json(source, target):
    """
    Transform bahaviors file in tsv to json for later evaluation
    Args:
        source:
        target:
    """
    behaviors = pd.read_table(
        source,
        header=None,
        names=['uid', 'time', 'clicked_news', 'impression'])
    f = open(target, "w")
    with tqdm(total=len(behaviors), desc="Transforming tsv to json") as pbar:
        for row in behaviors.itertuples(index=False):
            item = {}
            item['uid'] = row.uid[1:]
            item['time'] = row.time
            item['impression'] = {
                x.split('-')[0][1:]: int(x.split('-')[1])
                for x in row.impression.split()
            }
            f.write(json.dumps(item) + '\n')

            pbar.update(1)

    f.close()


if __name__ == '__main__':
    train_dir = '../dataset/train'
    test_dir = '../dataset/test'

    print('Process data for training')

    print('Clean up data')
    clean_dataset(path.join(train_dir, 'behaviors.tsv'),
                  path.join(train_dir, 'behaviors_cleaned.tsv'),
                  path.join(train_dir, 'news.tsv'),
                  path.join(train_dir, 'news_cleaned.tsv'))

    print('Balance data')
    balance(path.join(train_dir, 'behaviors_cleaned.tsv'),
            path.join(train_dir, 'behaviors_cleaned_balanced.tsv'), (1 / 2, 2))

    print('Parse news')
    parse_news(path.join(train_dir, 'news_cleaned.tsv'),
               path.join(train_dir, 'news_with_entity.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='train')

    print('Transform entity embeddings')
    transform_entity_embedding(path.join(train_dir, 'entity_embedding.vec'),
                               path.join(train_dir, 'entity_embedding.npy'),
                               path.join(train_dir, 'entity2int.tsv'))

    print('Transform word embeddings')
    transform_word_embedding(path.join(train_dir, 'glove.6B.100d.txt'),
                             path.join(train_dir, 'word_embedding.npy'),
                             path.join(train_dir, 'word2int.tsv'))

    print('\nProcess data for evaluation')

    print('Transform test data')
    transform2json(path.join(test_dir, 'behaviors.tsv'),
                   path.join(test_dir, 'truth.json'))

    print('Clean up data')
    clean_dataset(path.join(test_dir, 'behaviors.tsv'),
                  path.join(test_dir, 'behaviors_cleaned.tsv'),
                  path.join(test_dir, 'news.tsv'),
                  path.join(test_dir, 'news_cleaned.tsv'))

    print('Parse news')
    parse_news(path.join(test_dir, 'news_cleaned.tsv'),
               path.join(test_dir, 'news_with_entity.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='test')
