import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Reading the entire json metadata
import json
data  = []
with open("/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json", 'r') as f:
    for line in f:
        data.append(json.loads(line))

data = pd.DataFrame(data[:50000])

# importing necessary dependencies

from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

#Set up ElasticsearchDocumentStore
import os
from subprocess import Popen, PIPE, STDOUT
es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )

from haystack.database.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

#Use title column to pass as name and abstract column to pass as the text
#Write the dicts containing documents to our DB.
document_store.write_documents(data[['title', 'abstract']].rename(columns={'title':'name','abstract':'text'}).to_dict(orient='records'))

#Prepare Retriever, Reader & Finder
from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, context_window_size=500)
finder = Finder(reader, retriever)

#Test
sample_questions = ["What do we know about Bourin and Uchiyama?",
       "How is structure of event horizon linked with Morse theory?",
       "What do we know about symbiotic stars"]
prediction = finder.get_answers(question="What do we know about symbiotic stars", top_k_retriever=10, top_k_reader=2)
result = print_answers(prediction, details="minimal")