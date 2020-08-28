import numpy as np
import pandas as pd
from pandas import DataFrame
import sqlite3
from sqlalchemy import create_engine


# 1- Connect our python with SQLITE Dbase. If the DBASE doesnt exist, it will create one
conn = sqlite3.connect('airbnb.sqlite3')
curs = conn.cursor()


# 2- Create the table: airbnb
create1 = """
DROP TABLE IF exists airbnb;
CREATE TABLE airbnb  (
  Neighborhood  varchar(40),
  Bedrooms INT8, 
  Bathrooms  INT8,
  Beds  INT8,
  Accommodates  INT8,
  Guests_Included  INT8,
  Minimum_Nights  INT8,
  Maximum_Nights  INT8,
  Price  INT8 );
"""	


# 3a- Execute and commit after each changes
curs.executescript(create1) 
conn.commit()  

# 3b- Now the table is PREPARED in the DAtaBase

# 4- get the dataframe and update the column names
#     If you dont update the column name, it will follow whatever csv file gives you 
df = pd.read_csv('airbnb-listings.csv', usecols=[36, 51, 50, 52, 49,61, 63,64,56])
df.columns = ['Neighborhood', 'Bedrooms', 'Bathrooms', 'Beds', 'Accommodates', 'Guests_Included', 'Minimum_Nights', 'Maximum_Nights', 'Price' ]


# 5- Insert Pandas dataframe into SQLITE DataBase 
df.to_sql('airbnb', con = conn, if_exists='replace', index=False)

# Save connection to database
conn.commit() 


# 6- Now we can test on this python or using SQLITE apps.
a = conn.execute("""SELECT * FROM airbnb
	LIMIT 3""").fetchall()
print("\nshow some records only: \n" )
# [print(row) for row in a]
for row in a:
  print(row)


b = conn.execute("""
select count(*), Neighborhood from airbnb
where Bedrooms = 2
GROUP BY Neighborhood; 
"""  ).fetchall()

print("\n count of 2-bedrooms per neighborhood: \n")
[print(row) for row in b]


conn.cursor().execute(''' SELECT * from airbnb ''')
