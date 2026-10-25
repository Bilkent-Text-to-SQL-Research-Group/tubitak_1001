import pandas as pd
import numpy as np
import json
import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

json_file_path = os.path.expanduser("~/Desktop/Research Codes/train_tables.json")

with open(json_file_path, 'r') as file:
    data = json.load(file)

records = []

for db in data:
    db_id = db["db_id"]
    table_names = db["table_names"]
    column_names = db["column_names"]
    #dont take the first elemet of column_names
    column_names = column_names[1:]
    #add table names index zip them
    records.append({"db_id": db_id, "table_name": table_names, "column_name": column_names})

# Create a DataFrame
df = pd.DataFrame(records)

print(df.head())

corpus =  df['table_name'].apply(lambda x: " ".join(map(str, x)))
#add all the lists to a list
corpus = corpus.to_list()

deneme = corpus[0] + corpus[1]
print(deneme, "deneme")

#add all corpus
acorpus = " ".join(corpus)
#print(acorpus)
acorpus = word_tokenize(acorpus)
print(acorpus)
#tokenize the corpus
corpus = [word_tokenize(sentence) for sentence in corpus]


bm25models = {}
for i in range(len(corpus)):
    bm25 = BM25Okapi(corpus[i])
    bm25models[df["db_id"][i]] = bm25

allToks = BM25Okapi(acorpus)

#get the id of movies_4

index = df.loc[df['db_id'] == "car_retails"].index
print(index[0])
print(corpus[35])
scores = bm25models["car_retails"].get_top_n(["payments", "products"], corpus[35], n=4)
print(scores)

scores = allToks.get_top_n("payments", acorpus, n=100)
print(scores)