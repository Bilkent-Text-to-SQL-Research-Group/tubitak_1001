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
traintables_path = os.path.expanduser("~/Desktop/Research Codes/dev_dataset/dev_tables.json")
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

#>>>>>>BM25 Training for table names:

corpus_table =  database['table_name'].apply(lambda x: " ".join(map(str, x)))
#add all the lists to a list
corpus_table = corpus_table.to_list()

#add all corpus
acorpus_table = " ".join(corpus_table)
#print(acorpus)
acorpus_table = word_tokenize(acorpus_table)

#tokenize the corpus
corpus_table = [word_tokenize(sentence) for sentence in corpus_table]

# Train a model for each database id!
bm25models = {}
for i in range(len(database["db_id"])):
    bm25 = BM25Okapi(corpus_table[i])
    bm25models[database["db_id"][i]] = bm25

general_bm25 = BM25Okapi(acorpus_table)

#>>>>>>BM25 Training for column names:

corpus_column =  database['column_name'].apply(lambda x: " ".join(map(str, x)))
#add all the lists to a list
corpus_column = corpus_column.to_list()

#add all corpus
acorpus_column = " ".join(corpus_column)
#print(acorpus)
acorpus_column = word_tokenize(acorpus_column)

#tokenize the corpus
corpus_column = [word_tokenize(sentence) for sentence in corpus_column]

# Train a model for each database id!
bm25models_column = {}
for i in range(len(database["db_id"])):
    bm25_column = BM25Okapi(corpus_column[i])
    bm25models_column[database["db_id"][i]] = bm25_column

general_bm25_column = BM25Okapi(acorpus_column)

#>>>>>>>>> Load the model to get similar words for column names
model_path = os.path.expanduser("~/Desktop/Research Codes/GoogleNews-vectors-native.kv")
word_vectors = KeyedVectors.load(model_path)

# Get the feedbacks to ruin
json_file_path = os.path.expanduser("~/Desktop/Research Codes/dev_feedback.json")

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
        "feedback": entry.get("intermediate_step"),
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

# Delete a table and its columns for "Missing table" feedback
def delete_table(feedback):
    tables, columns, values = parse_feedback(feedback)
    tableNo = len(tables)
    if(tableNo == 1):
        return None
    indx = np.random.randint(0,tableNo)
    alias = tables[f"table{indx}"]["alias"]
    indexes = []
    deleted_tables = []
    if tables[f"table{indx}"]["alias"] is None:
        deleted_tables.append(tables[f"table{indx}"]["name"])
        del tables[f"table{indx}"]
    else:
        deleted_tables.append(tables[f"table{indx}"]["name"])
        del tables[f"table{indx}"]
        for column in columns:
            if columns[column]["table_name"] == alias:
                indexes.append(column)
    for index in indexes:
        del columns[index]
    result_feedback = dic2feedback(tables, columns, values)
    return result_feedback, deleted_tables

# Insert one or more table to the intermediate step for "Redundant table" feedback
def insert_false_table(feedback, db_id):
    tables, columns, values = parse_feedback(feedback)
    tableNo = len(tables)

    #Retrieve the wanted database's all tables
    idx12 =  database.loc[database['db_id'] == db_id].index
    idx = idx12[0]
    table_names = database["table_name"][idx]

    # Get the tableX's X from current intermediate step. (Do this because a table might be deleted)
    remaining_tables = list(tables.keys())
    remaining_idxs = [int(key[5:]) for key in tables.keys()]
    # We will choose a new table similar to this index's table name
    indx = random.choice(remaining_idxs)


    #indx = np.random.randint(0,tableNo)
    # Get the used tables in order to get the largest alias no
    used_tables = []
    alias = None #CHANGED
    for table in tables:
        used_tables.append(tables[table]["name"])
        if tables[table]["alias"] is not None:
            alias = tables[table]["alias"]

    # If alias used, get the number of it since the train aliases are all like T1, T3 etc.
    used_alias = None
    if alias is not None:
        #alias number
        a1parse = re.match(r"T([0-9]+)", alias)
        if a1parse:
            # Extract the captured digits
            alias = a1parse.group(1)
            used_alias = int(alias)
    #Find how many tables are used to see the unused tables
    unique_tables = list(set(used_tables))
    unique_tables.sort()
    used_tableNo = len(unique_tables)
    #This many tables are not used in the feedback
    alt_table = abs(used_tableNo - len(database["table_name"][idx]))

    # Store inserted tables' names to give feedback!
    inserted_tables = []
    #If all the tables are used insert a random table similar to this domain or add a previously used table
    if alt_table == 0:
        text = None
        scores = general_bm25.get_top_n(tables[f"table{indx}"]["name"], acorpus_table, n=5)
        for score in scores:
            if score not in used_tables:
                text = score
                break
        # Add a random table from db_id
        if text is None:
            text = unique_tables[np.random.randint(0,used_tableNo)]
        
        #Check alias
        alias1 = None
        if used_alias is not None:
            alias1 = f"T{used_alias + 1}"

        inserted_tables.append(text)
        #Insert
        tables[f"table{tableNo}"] = {"name": text, "alias": alias1}
        result_feedback = dic2feedback(tables, columns, values)
        return result_feedback, inserted_tables
    else: #If all the tables are not used yet

        #Randomly decide how many tables to insert (can be multiple of them)
        addNo = np.random.randint(0,alt_table)
        if addNo == 0:
            addNo = 1

        #Find unused tables, choose a random table among them and insert it
        available_tables = []
        for table in table_names:
            if table not in used_tables:
                available_tables.append(table)
        
        #Insert addNo many tables to the intermediate step
        for i in range(0, addNo):
            text = available_tables[np.random.randint(0,len(available_tables))]
            alias1 = None
            if used_alias is not None:
                alias1 = f"T{used_alias + 1 + i}"
            inserted_tables.append(text)
            tables[f"table{tableNo + i}"] = {"name": text, "alias": alias1}
    #Create the new structure and return
    result_feedback = dic2feedback(tables, columns, values)
    return result_feedback, inserted_tables

# Try to change a table name with its similar word for "Incorrect table" feedback
def change_table(feedback, db_id):
    # Parse the intermediate step and get the tables, columns, and values. We will only use tables.
    tables, columns, values = parse_feedback(feedback)
    tableNo = len(tables)

    #Get all tables taht the database has
    idx12 =  database.loc[database['db_id'] == db_id].index
    idx = idx12[0]
    table_names = database["table_name"][idx]

    #Randomly choose the table that we are going to change
    indx = np.random.randint(0,tableNo)
    changed_table = tables[f"table{indx}"]["name"]

    #Get the used tables to not change a table to a already used table.
    used_tables = []
    for table in tables:
        used_tables.append(tables[table]["name"])
    text = changed_table
    try:
        # Get the most similar 100 words to choosen table's name. If there is a unused table of db_id in there, change the table with it.
        tops = general_bm25.get_top_n(changed_table, acorpus_table, n=100)
        for top in tops:
            if top in table_names and top not in used_tables:
                text = top
                break
        #If the table's structure liek word1_word2, change the table as word1 since it is very similar to the previous one mostly.
        text1 = processText(text)
        #Important!!!!!!!
        if text1 != text:
            text = text1
        else:
            #If the structure is not like word1_word2, try to find a similar word with GoogleNews-vectors-native model.
            text = word_vectors.most_similar(text.lower(), topn=3)[np.random.randint(0, 3)][0]
    except:
        #If nothing works, try to find the most similar table among the db_id's tables.
        #Not recommended since the most of the databases table name corpus is not large enough for a accuracte similartiy calculation.
        scores = bm25models[db_id].get_top_n(changed_table, corpus_table[idx], n=2)
        text = scores[1]
        #If found table name is used, use it again. It's fine. It won't have an impact on the model training since it is a pretty rare case.
    changed_table = tables[f"table{indx}"]["name"]
    new_table = text
    tables[f"table{indx}"]["name"] = text
    result_feedback = dic2feedback(tables, columns, values)
    return result_feedback, changed_table, new_table

# Remove some columns from the feedback
def delete_column(feedback):
    tables, columns, values = parse_feedback(feedback)
    columnNo = len(columns)
    if(columnNo == 1):
        return None
    indx = np.random.randint(0,columnNo)
    deleted_columns = []
    deleted_indxs = []
    for i in range(0, indx):
        #pick a random indx that is not in deleted_indxs
        del_indx = np.random.randint(0,columnNo)
        while del_indx in deleted_indxs:
            del_indx = np.random.randint(0,columnNo)
        deleted_columns.append(columns[f"column{del_indx}"]["column_name"])  #ERRRORRR
        deleted_indxs.append(del_indx)
        del columns[f"column{del_indx}"]
    result_feedback = dic2feedback(tables, columns, values)
    return result_feedback, deleted_columns

def insert_false_column(feedback, db_id):
    #Retrieve the elements
    tables, columns, values = parse_feedback(feedback)
    columnNo = len(columns)
    #Retrieve the wanted database's all columns
    idx12 =  database.loc[database['db_id'] == db_id].index
    idx = idx12[0]
    column_names = database["column_name"][idx]

    # Get the columnX's X from current intermediate step. (Do this because a column might be deleted)
    remaining_idxs = [int(key[6:]) for key in columns.keys()]
    # We will choose a new column similar to this index's table name
    indx = random.choice(remaining_idxs)

    #How many columns are used in the feedback (no duplicates)
    used_columns = []
    for column in columns:
        if columns[column]["column_name"] not in used_columns:
            used_columns.append(columns[column]["column_name"])

    #How many columns are NOT used in the feedback
    alt_column = abs(len(column_names) - len(used_columns))
    available_columns = []
    for column in column_names:
        if column not in used_columns:
            available_columns.append(column)

    inserted_columns = []
    if alt_column == 0:
        text = None
        scores = general_bm25_column.get_top_n(columns[f"column{indx}"]["column_name"], acorpus_column, n=5)
        for score in scores:
            if score not in used_columns:
                text = score
                break
        # Add a random column from db_id
        if text is None:
            text = column_names[np.random.randint(0,len(column_names))] #duplicate a column
        inserted_columns.append(text)
        columns[f"column{columnNo}"] = {"table_name": None, "column_name": text}
        result_feedback = dic2feedback(tables, columns, values)
        return result_feedback, inserted_columns
    else:
        addNo = np.random.randint(0,alt_column)
        if addNo == 0:
            addNo = 1
        for i in range(0, addNo):
            text = available_columns[np.random.randint(0,len(available_columns))]
            inserted_columns.append(text)
            available_columns.remove(text)
            columns[f"column{columnNo + i}"] = {"table_name": None, "column_name": text}
        #Create the new structure and return
        result_feedback = dic2feedback(tables, columns, values)
        return result_feedback, inserted_columns

def change_column(feedback, db_id):
    # Parse the intermediate step and get the tables, columns, and values. We will only use tables.
    tables, columns, values = parse_feedback(feedback)
    columnNo = len(columns)

    #Get all column taht the database has
    idx12 =  database.loc[database['db_id'] == db_id].index
    idx = idx12[0]
    column_names = database["column_name"][idx]

    indx = np.random.randint(0,columnNo)
    changed_column = columns[f"column{indx}"]["column_name"]

    used_columns = []
    for column in columns:
        if columns[column]["column_name"] not in used_columns:
            used_columns.append(columns[column]["column_name"])

    text = changed_column
    try:
        # Get the most similar 100 words to choosen column's name. If there is a unused column of db_id in there, change the column with it.
        tops = general_bm25_column.get_top_n(text, acorpus_column, n=100)
        for top in tops:
            if top in column_names and top not in used_columns:
                text = top
                break
        #If the column's structure liek word1_word2, change the column as word1 since it is very similar to the previous one mostly.
        text1 = processText(text)
        #Important!!!!!!!
        if text1 != text:
            text = text1
        else:
            #If the structure is not like word1_word2, try to find a similar word with GoogleNews-vectors-native model.
            text = word_vectors.most_similar(text.lower(), topn=3)[np.random.randint(0, 3)][0]
    except:
        #If nothing works, try to find the most similar column among the db_id's columns.
        #Not recommended since the most of the databases table name corpus is not large enough for a accuracte similartiy calculation.
        scores = bm25models_column[db_id].get_top_n(changed_column, corpus_column[idx], n=2)
        text = scores[1]
        #If found column name is used, use it again. It's fine. It won't have an impact on the model training since it is a pretty rare case.
    changed_column = columns[f"column{indx}"]["column_name"]
    new_column = text
    columns[f"column{indx}"]["column_name"] = text
    result_feedback = dic2feedback(tables, columns, values)
    return result_feedback, changed_column, new_column


ruined = []
delete_count = 0
sa = 0
for i in range(df.shape[0]):
    sa += 1
    if sa % 100 == 0:
        print("Progress: ", sa, "/", df.shape[0])
    count = 0
    row = {
        "query_id": df['query_id'][i],
        "feedback_id": f"{0}", #true feedbacks' ids are 0
        "db_id": df['db_id'][i],
        "question": df['question'][i],
        "evidence": df['evidence'][i],
        "SQL": df['SQL'][i],
        "inter_Step": df['feedback'][i],
        "false_inter_step": None,
        "feedback_line": "Ground Truth",
        "operation": "True intermediate Step",
    }
    ruined.append(row)
    count += 1

    f1, ct, nt = change_table(df["feedback"][i], df["db_id"][i])
    row = {
        "query_id": df['query_id'][i],
        "feedback_id": f"{count}",
        "db_id": df['db_id'][i],
        "question": df['question'][i],
        "evidence": df['evidence'][i],
        "SQL": df['SQL'][i],
        "inter_Step": df['feedback'][i],
        "false_inter_step": f1,
        "feedback_line": "incorrect " f"{nt}",
        "operation": "change",
    }
    count += 1
    ruined.append(row)


    k_line, it = insert_false_table(df["feedback"][i], df["db_id"][i])
    row = {
        "query_id": df['query_id'][i],
        "feedback_id": f"{count}", #true feedbacks' ids are 0
        "db_id": df['db_id'][i],
        "question": df['question'][i],
        "evidence": df['evidence'][i],
        "SQL": df['SQL'][i],
        "inter_Step": df['feedback'][i],
        "false_inter_step": k_line,
        "feedback_line": "redundant " f"{it}",
        "operation": "insertion",
    }
    count += 1
    ruined.append(row)


    if df["table_count"][i] > 1:
        d_line, dt = delete_table(df["feedback"][i])
        row = {
            "query_id": df['query_id'][i],
            "feedback_id": f"{count}", #true feedbacks' ids are 0
            "db_id": df['db_id'][i],
            "question": df['question'][i],
            "evidence": df['evidence'][i],
            "SQL": df['SQL'][i],
            "inter_Step": df['feedback'][i],
            "false_inter_step": d_line,
            "feedback_line": "missing " f"{dt}",
            "operation": "deletion",
        }
        delete_count += 1
        count += 1
        ruined.append(row)

    """if df["column_count"][i] > 1:
        f1, ct, nt = change_column(df["feedback"][i], df["db_id"][i])
        row = {
            "query_id": df['query_id'][i],
            "feedback_id": f"{count}", #true feedbacks' ids are 0
            "db_id": df['db_id'][i],
            "question": df['question'][i],
            "evidence": df['evidence'][i],
            "SQL": df['SQL'][i],
            "inter_Step": df['feedback'][i],
            "false_inter_step": f1,
            "feedback_line": "incorrect " f"{nt}",
            "operation": "change column",
        }
        count += 1
        ruined.append(row)

    if df["column_count"][i] > 1:
        k_line, it = insert_false_column(df["feedback"][i], df["db_id"][i])
        row = {
            "query_id": df['query_id'][i],
            "feedback_id": f"{count}", #true feedbacks' ids are 0
            "db_id": df['db_id'][i],
            "question": df['question'][i],
            "evidence": df['evidence'][i],
            "SQL": df['SQL'][i],
            "inter_Step": df['feedback'][i],
            "false_inter_step": k_line,
            "feedback_line": "redundant " f"{it}",
            "operation": "column insertion",
        }
        count += 1
        ruined.append(row)

    if df["column_count"][i] > 1:
        d_line, dt = delete_column(df["feedback"][i])
        row = {
            "query_id": df['query_id'][i],
            "feedback_id": f"{count}", #true feedbacks' ids are 0
            "db_id": df['db_id'][i],
            "question": df['question'][i],
            "evidence": df['evidence'][i],
            "SQL": df['SQL'][i],
            "inter_Step": df['feedback'][i],
            "false_inter_step": d_line,
            "feedback_line": "missing " f"{dt}",
            "operation": "column deletion",
        }
        delete_count += 1
        count += 1
        ruined.append(row)"""
    
neww = pd.DataFrame(ruined)
#convert ruined array to json file and save
json_output = neww.to_dict(orient="records")

# Save the JSON to a file
file_path = "train_inter.json"
with open(file_path, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

memory_in_bytes = neww.memory_usage(deep=True).sum()

# Convert to KB, MB, or GB
memory_in_kb = memory_in_bytes / 1024
memory_in_mb = memory_in_kb / 1024
memory_in_gb = memory_in_mb / 1024

# Print memory usage
print(f"Memory usage: {memory_in_bytes} bytes")
print(f"Memory usage: {memory_in_kb:.2f} KB")
print(f"Memory usage: {memory_in_mb:.2f} MB")
print(f"Memory usage: {memory_in_gb:.6f} GB")

print("Total ground truth intermediate steps:", df.shape[0])
print("Total deletion operations:", delete_count)
count = float(count)
print("Inter steps per query:", count/df.shape[0])