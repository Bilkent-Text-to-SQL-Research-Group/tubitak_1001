import pandas as pd
import numpy as np
import json
import os
import re
import random
from sqlglot import Tokenizer, parse_one, expressions
from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

#First, prepare the data to train bm25 model. (To get similar table names while changing)
traintables_path = os.path.expanduser("~/Desktop/Research Codes/research/train_tables.json")
with open(traintables_path, 'r') as file:
    train_data = json.load(file)

records = []

for db in train_data:
    db_id = db["db_id"]
    table_names = db["table_names"]
    column_names = db["column_names"]
    #dont take the first elemet of column_names
    column_names = column_names[1:]
    #add table names index zip them
    records.append({"db_id": db_id, "table_name": table_names, "column_name": column_names})

# Create a DataFrame
database = pd.DataFrame(records)

#Check db number
print(len(database["db_id"]))

# Get the feedbacks to ruin
json_file_path = os.path.expanduser("~/Desktop/Research Codes/train_feedback.json")

with open(json_file_path, 'r') as file:
    data = json.load(file)

query_ids = []
# Initialize lists to store the extracted data
rows = []

# Single iteration to collect the required fields
for i, entry in enumerate(data):
    row = {
        "query_id": entry.get("query_id"),
        "feedback_id": f"{0}", #true feedbacks' ids are 0
        "db_id": entry.get("db_id"),
        "question": entry.get("question"),
        "evidence": entry.get("evidence"),
        "SQL": entry.get("SQL"),
        "feedback": entry.get("feedback"),
        "table_count": entry.get("table_count"),
        "column_count": entry.get("column_count"),
        "value_count": entry.get("value_count"),
        "feedback_line": "",
    }
    rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows)

print(df.shape) # Output: (9428, 6)

#
# Input: feedback (string)
# Output: tables (dict), columns (dict), values (dict)
#
global val_check
val_check = 0
def parse_feedback(feedback):
    # Initialize dictionaries to store parsed data
    tables = {}
    columns = {}
    values = {}

    # Parse the input line by line. Can be only 1 instead of table1.
    for line in feedback.splitlines():
        c = 0
        if line.startswith("#table"):
            #If the line include AS keyword, it means that the table is aliased.
            if " AS " in line:
                if "\"" in line:
                    match_with_alias = re.match(r"#table(\d+)\s+\"([^\"]+)\"\s+AS\s+(\S+)", line) #to handle this: "#table0 \"Sales Orders\" AS T1\n#table1 \"Sales Team\" AS T2\n
                    table_id, table_name, alias = match_with_alias.groups()
                    tables[f"table{table_id}"] = {"name": table_name, "alias": alias}
                else:
                    # Extract table ID and name
                    match_with_alias = re.match(r"#table(\d+)\s+(\S+)\s+AS\s+(\S+)", line)
                    table_id, table_name, alias = match_with_alias.groups() 
                    tables[f"table{table_id}"] = {"name": table_name, "alias": alias}
            else:
                match_without_alias = re.match(r"#table(\d+)\s+(\S+)", line)
                table_id, table_name = match_without_alias.groups()
                tables[f"table{table_id}"] = {"name": table_name, "alias": None}
        elif line.startswith("#column"):
            #If the line includes . keyword, it means that the column name has a table name in front of it.
            if "." in line:
                match_with_alias = re.match(r"#column(\d+)\s+(\S+)\.(\S+)", line)
                column_id, table_alias, column_name = match_with_alias.groups()
                columns[f"column{column_id}"] = {"table_name": table_alias, "column_name": column_name}
            else:
                match_without_alias = re.match(r"#column(\d+)\s+(\S+)", line)
                column_id, column_name = match_without_alias.groups()
                columns[f"column{column_id}"] = {"table_name": None, "column_name": column_name}
        elif line.startswith("#value"):
            # Extract value ID and value
            match = re.match(r"#value(\d+)\s+(.+)", line)
            if match:
                global val_check
                val_check += 1
                value_id, value = match.groups()
                values[f"value{value_id}"] = value

    return tables, columns, values


def processText(text):
    a1 = text
    if "_" in text:
        text = re.match(r"([a-zA-Z]+)([-_])([a-zA-Z]+)", text)
        a1 = text.groups()[0]
    return a1

def processColumn(column):
    a1 = column
    print(column)
    # parse column name
    a1parse = re.match(r"([a-z]+)([A-Z][a-zA-Z]*)*", a1)
    if a1parse is not None:
        a1 = a1parse.groups()[0]
    print(a1)
    return a1


def dic2feedback(tables, columns, values):
    feedback = ""
    for table in tables:
        if tables[table]["alias"] is not None:
            feedback += f"#{table} {tables[table]['name']} AS {tables[table]['alias']}\n"
        else:
            feedback += f"#{table} {tables[table]['name']}\n"
    for column in columns:
        if columns[column]["table_name"] is not None:
            feedback += f"#{column} {columns[column]['table_name']}.{columns[column]['column_name']}\n"
        else:
            feedback += f"#{column} {columns[column]['column_name']}\n"
    for value in values:
        feedback += f"#{value} {values[value]}\n"
    return feedback


# now ruin feedbacks
ruined = []
count = 0
for i in range(df.shape[0]):
    row = {
        "query_id": df['query_id'][i],
        "feedback_id": f"{0}", #true feedbacks' ids are 0
        "db_id": df['db_id'][i],
        "question": df['question'][i],
        "evidence": df['evidence'][i],
        "SQL": df['SQL'][i],
        "inter_step": df['feedback'][i],
        "false_inter_step": None,
        "feedback_line": "NONE",
        "operation": "True intermediate Step",
    }
    ruined.append(row)
    count += 1
    
neww = pd.DataFrame(ruined)
#convert ruined array to json file and save
json_output = neww.to_dict(orient="records")

# Save the JSON to a file
file_path = "ground_truth_inters.json"
with open(file_path, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

print("Total ground truth intermediate steps:", df.shape[0])
print("Total feedbacks:", count)