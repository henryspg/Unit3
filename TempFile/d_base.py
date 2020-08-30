
import csv
from dotenv import load_dotenv
import psycopg2
import json
from psycopg2.extras import execute_values

load_dotenv()  # loads contents of the .env file into the script's environment

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

print(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST)

# Connect to SQL-hosted PostgreSQL
connection = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

print("CONNECTION", connection)
# A "cursor", a structure to iterate over db records to perform queries
cursor = connection.cursor()
print("CURSOR", cursor)

sql_create = """DROP TABLE IF EXISTS
airbnb_table;
  CREATE TABLE airbnb_table (
  id        SERIAL PRIMARY KEY,
  Accomodates int,
  Bathrooms int,
  Bedrooms int,
  Beds int,
  Guests Included int,
  Minimum Nights,
  Maximum Nights

);
"""
# Avoiding the table already exists error
cursor.execute(sql_create)

# Avoiding the table already exists error
connection.commit()

# Assign the titanic csv to a reader object
reader = list(csv.reader(open("airbnb-listings.csv", "r")))

# Record count

record_count = 0

# Insert information into the database
for row in reader[1:]:

    sqlInsert = '''INSERT INTO airbnb_table (Accomodates, Bathrooms, Bedrooms, Beds, Guests Included, Minimum nights, Maximun nights)
    VALUES(%s, %s, %s, %s, %s, %s, %s) '''

   # Execute query and commit changes.
    cursor.execute(sqlInsert, (row[0],
                               row[1],
                               row[2],
                               row[3],
                               row[4],
                               row[5],
                               row[6],
                               ))

    # Increment the record count.
    record_count += 1


connection.commit()
