import json
import pandas as pd
from haystack import Finder
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever


data = []
with open('../archive/arxiv-metadata-oai-snapshot.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

data = pd.DataFrame(data[:50000])

# data = pd.read_csv('test.txt', sep='\t')
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
document_store.write_documents(
    data[['title', 'abstract']].rename(columns={'title': 'name', 'abstract': 'text'}).to_dict(orient='records'))

retriever = ElasticsearchRetriever(document_store=document_store)
reader = TransformersReader(model_name_or_path='deepset/roberta-base-squad2',
                            tokenizer='deepset/roberta-base-squad2',
                            context_window_size=500, use_gpu=-1)
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, context_window_size=500)
finder = Finder(reader, retriever)

if __name__ == '__main__':
    # questions = ["What do we know about Bourin and Uchiyama?"]
    '''
    prediction = finder.get_answers(question="What do we know about symbiotic stars?",
                                    top_k_retriever=10, top_k_reader=3)
    print_answers(prediction, details='minimal')
    '''
    while True:
        qes = input('Question: ')
        # print(qes)
        prediction = finder.get_answers(question=qes, top_k_retriever=5, top_k_reader=5)
        print_answers(prediction, details='minimal')
