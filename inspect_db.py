import sqlite3

connection = sqlite3.connect("data/mini_ids.db")

print("Tables in the database:")
tables = connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

for (table_name,) in tables:
    print(f"\nTable: {table_name}")

    columns = connection.execute(
        f"PRAGMA table_info('{table_name}')"
    ).fetchall()

    for column in columns:
        print(column)

connection.close()
