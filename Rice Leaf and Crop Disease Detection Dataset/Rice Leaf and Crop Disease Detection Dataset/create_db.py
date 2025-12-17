# create_db.py

import sqlite3

def create_database():
    # Connect to (or create) the database file
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Create users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database 'users.db' created successfully with table 'users'.")

if __name__ == "__main__":
    create_database()
