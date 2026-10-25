#Convert sql's to feedback format and store as a json file named train_feedback.json

import pandas as pd
import numpy as np
import json
import os
from sqlglot import Tokenizer, parse_one, expressions

json_file_path = os.path.expanduser("~/Desktop/Research Codes/dev_dataset/dev.json")

with open(json_file_path, 'r') as file:
    data = json.load(file)

query_ids = []
# Initialize lists to store the extracted data
rows = []

# Single iteration to collect the required fields
for i, entry in enumerate(data):
    row = {
        "query_id": i,
        "db_id": entry.get("db_id"),
        "question": entry.get("question"),
        "evidence": entry.get("evidence"),
        "SQL": entry.get("SQL"),
    }
    rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows)

print(df.head())


# Create a pandas DataFrame
sql_df = df[["SQL"]].copy()

print(sql_df.head())

#First convert all ` to '
def convert_to_single_quote(sql):
    return sql.replace("`","'")

sql_df["SQL"] = sql_df["SQL"].apply(convert_to_single_quote) #Otherwise 1000s of errors will be generated


def parseSQL(sql):
    try:
        Tokenizer().tokenize(sql)
        #Parse it.
        prs = parse_one(sql)
        #print(repr(prs))
        #All retrieved tables (from)
        all_tables = [table.sql() for table in prs.find_all(expressions.Table)]
        table_count = len(all_tables)
        #All used attributes
        all_columns = [Column.sql() for Column in prs.find_all(expressions.Column)]
        column_count = len(all_columns)
        #All values
        all_values = [Value.sql() for Value in prs.find_all(expressions.Literal)]
        value_count = len(all_values)
        #Columns used in select clause
        select_columns = [
            column.sql()
            for select in prs.find_all(expressions.Select)
            for column in select.expressions
            if isinstance(column, expressions.Column)
        ]
        #Columns used in where clause
        where_columns = [
            column.sql()
            for where in prs.find_all(expressions.Where)
            for column in where.find_all(expressions.Column)
        ]

        #Is limit used or not????
        limit = [limit.sql() for limit in prs.find_all(expressions.Limit)] #[] if none, [LIMIT 1] if LIMIT 1
        #All used aggregate functions in the query
        aggregate_classes = [expressions.Avg, expressions.Sum, expressions.Count, expressions.Max, expressions.Min]
        aggregate_functions = [
            expr.sql()
            for cls in aggregate_classes
            for expr in prs.find_all(cls)
        ]

        """print("All Tables Used:", all_tables)
        print("All Columns Used:", all_columns)
        print("All Values Used:", all_values)
        print("All aggregates:", aggregate_functions)
        print("Select clause's columns:",select_columns)
        print("Where clause's columns:",where_columns)
        print("Limit:",limit)"""

        table_output = "\n".join(f"#table{i} {table}" for i, table in enumerate(all_tables))
        column_output = "\n".join(f"#column{i} {column}" for i, column in enumerate(all_columns))
        value_output = "\n".join(f"#value{i} {value}" for i, value in enumerate(all_values))

        combine = f"{table_output}\n{column_output}\n{value_output}"
        #print(combine)

        return combine, table_count, column_count, value_count
    except:
        return "Error"

# Parse the SQL queries
df["intermediate_step"], df["table_count"], df["column_count"], df["value_count"] = zip(*sql_df["SQL"].apply(parseSQL))
#sql_df["parsed"] = df["feedback"]



#output_csv_path = "sql_df_output.csv"  # You can modify this path as needed

json_output = df.to_dict(orient="records")

# Save the JSON to a file
file_path = "dev_feedback.json"
with open(file_path, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

# Save the DataFrame to a CSV file
#sql_df.to_csv(output_csv_path, index=False)