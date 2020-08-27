
# Taken from:  https://www.w3resource.com/pandas/dataframe/dataframe-to_sql.php

# Examples
# Create an in-memory SQLite database:

# import numpy as np
# import pandas as pd

# from sqlalchemy import create_engine
# engine = create_engine('sqlite://', echo=False)

# df = pd.DataFrame({'name' : ['User P', 'User Q', 'User R']})
# df


# 	name
# 0	User P
# 1	User Q
# 2	User R


# df.to_sql('users', con=engine)  ## 'users' is the table name in sql
# engine.execute("SELECT * FROM users").fetchall()

# #[(0, 'User P'), (1, 'User Q'), (2, 'User R')]


# Create a table from scratch with 3 rows:

# df1 = pd.DataFrame({'name' : ['User S', 'User T']})
# df1.to_sql('users', con=engine, if_exists='append')
# engine.execute("SELECT * FROM users").fetchall()

# # [(0, 'User P'), (1, 'User Q'), (2, 'User R'), (0, 'User S'), (1, 'User T')]

# def connect_to_db(db_name='rpg_db.sqlite3'):
#     return sqlite3.connect(db_name)


# def execute_query(cursor, query):
#     cursor.execute(query)
#     return cursor.fetchall()


# GET_CHARACTERS = """
#   SELECT *
#   FROM charactercreator_character;
# """


import numpy as np
import pandas as pd
from pandas import DataFrame
import sqlite3
from sqlalchemy import create_engine


# Taken from:  https://www.w3resource.com/pandas/dataframe/dataframe-to_sql.php

# Examples
# Create an in-memory SQLite database:

# import numpy as np
# import pandas as pd

# from sqlalchemy import create_engine
# engine = create_engine('sqlite://', echo=False)

# df = pd.DataFrame({'name' : ['User P', 'User Q', 'User R']})
# df


# 	name
# 0	User P
# 1	User Q
# 2	User R


# df.to_sql('users', con=engine)  ## 'users' is the table name in sql
# engine.execute("SELECT * FROM users").fetchall()

# #[(0, 'User P'), (1, 'User Q'), (2, 'User R')]


# Create a table from scratch with 3 rows:

# df1 = pd.DataFrame({'name' : ['User S', 'User T']})
# df1.to_sql('users', con=engine, if_exists='append')
# engine.execute("SELECT * FROM users").fetchall()

# # [(0, 'User P'), (1, 'User Q'), (2, 'User R'), (0, 'User S'), (1, 'User T')]

# def connect_to_db(db_name='rpg_db.sqlite3'):
#     return sqlite3.connect(db_name)


# def execute_query(cursor, query):
#     cursor.execute(query)
#     return cursor.fetchall()


# GET_CHARACTERS = """
#   SELECT *
#   FROM charactercreator_character;
# """


import numpy as np
import pandas as pd
from pandas import DataFrame
import sqlite3
from sqlalchemy import create_engine


# 1- Connect our python with SQLITE Dbase. If the DBASE doesnt exist, it will create one
conn = sqlite3.connect('airbnb.sqlite3')
curs = conn.cursor()


# 2- Create the table: ‘TITANIC’
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
df = pd.read_csv('AbnbSF_clean.csv', usecols=[0, 4,5,6,7,8,9,11,12])
df.columns = ['Neighborhood', 'Bedrooms', 'Bathrooms', 'Beds', 'Accommodates', 'Guests_Included', 'Minimum_Nights', 'Maximum_Nights', 'Price' ]


# 5- Insert Pandas dataframe into SQLITE DataBase 
df.to_sql('airbnb', con = conn, if_exists='replace', index=False)

# Save connection to database
conn.commit() 


# 6- Now we can test on this python or using SQLITE apps.
a = conn.execute("""SELECT * FROM airbnb
	LIMIT 3""").fetchall()
print("\nshow some records only: \n" )
[print(row) for row in a]


b = conn.execute("""
select count(*), Neighborhood from airbnb
where Bedrooms = 2
GROUP BY Neighborhood; 
"""  ).fetchall()

print("\n count of 2-bedrooms per neighborhood: \n")
[print(row) for row in b]


conn.cursor().execute(''' SELECT * from airbnb ''')

# 7- How to create DataFrame (df2) from the table in the dbase?
    # this result is empty
# dn = DataFrame(conn.cursor().fetchall(), columns=['Survived', 'Pclass', 'Age'] )
# print(dn)
