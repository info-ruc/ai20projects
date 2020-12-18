import os
import psutil
import logging
from simpletransformers.t5 import T5Model
from simpletransformers.seq2seq import Seq2SeqModel

import numpy as np
import json
import pandas as pd


def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB: ' + str(np.round(memory_use, 2))


def get_metadata():
    path = '../archive/arxiv-metadata-oai-snapshot.json'
    with open(path, 'r') as f:
        for line in f:
            yield line


if __name__ == '__main__':
    metadata = get_metadata()
    titles = []
    abstracts = []
    years = []
    for paper in metadata:
        paper_dict = json.loads(paper)
        ref = paper_dict.get('journal-ref')
        try:
            year = int(ref[-4:])
            if 2016 < year < 2021:
                years.append(year)
                titles.append(paper_dict.get('title'))
                abstracts.append(paper_dict.get('abstract'))
        except:
            pass

    print(len(titles), len(abstracts), len(years))

    papers = pd.DataFrame({
        'title': titles,
        'abstract': abstracts,
        'year': years
    })
    print(papers.head())

    del titles, abstracts, years
    print(cpu_stats())

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    papers = papers[['title', 'abstract']]
    papers.columns = ['target_text', 'input_text']
    papers = papers.dropna()

    eval_df = papers.sample(frac=0.2, random_state=101)
    train_df = papers.drop(eval_df.index)
    print(train_df.shape, eval_df.shape)

    train_df['prefix'] = "summarize"
    eval_df['prefix'] = 'summarize'

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 512,
        "train_batch_size": 16,
        "num_train_epochs": 4,
    }

    # model = Seq2SeqModel(encoder_decoder_type="bart",
                         # encoder_decoder_name="facebook/bart-base",
                         # args=model_args)
    model = T5Model("t5-small", args=model_args, use_cuda=True)
    model.train_model(train_df)
    results = model.eval_model(eval_df)
    print(results)

    random_num = 350
    actual_title = eval_df.iloc[random_num]['target_text']
    actual_abstract = ["summarize: " + eval_df.iloc[random_num]['input_text']]
    predicted_title = model.predict(actual_abstract)

    print(f'Actual Title: {actual_title}')
    print(f'Predicted Title: {predicted_title}')
    print(f'Actual Abstract: {actual_abstract}')
