import os
import sqlite3
import json

# construct a path to wherever your database exists
#DB_FILEPATH = "demo_data.sqlite3.db"
DB_FILEPATH = os.path.join(os.path.dirname(__file__), "airbnb_data.sqlite3")

connection = sqlite3.connect(DB_FILEPATH)
#print("CONNECTION:", connection)

# Avoiding the table already exists error
connection.commit()

print("CONNECTION", connection)
# A "cursor", a structure to iterate over db records to perform queries
cursor = connection.cursor()
print("CURSOR", cursor)

#sql_create = """CREATE TABLE Airbnb_ Listings """

sql_create = """DROP TABLE IF EXISTS demo; CREATE TABLE airbnb_listings (Accomodates int,
  Bathrooms float,
  Bedrooms int,
  Beds int,
  Guests Included int,
  Minimum Nights int,
  Maximum Nights int)"""

connection.commit()


# Assign the airbnb listings csv to a reader object
reader = list(csv.reader(open("airbnb_listings.csv", "r")))

# Record count

record_count = 0

# Insert information into the database
for row in reader[1:]:

    sqlInsert = '''INSERT INTO airbnb_listings (Accomodates, Bathrooms, Bedrooms, Beds, Guests Included, Minimum nights, Maximun nights)
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
