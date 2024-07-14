import sqlite3

class Database:
    def __init__(self, db_name="hypertune.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                prompt TEXT,
                output TEXT,
                temperature REAL,
                top_p REAL,
                score REAL
            )
        """)

    def insert_result(self, prompt, output, temperature, top_p, score):
        self.conn.execute("""
            INSERT INTO results (prompt, output, temperature, top_p, score)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt, output, temperature, top_p, score))
        self.conn.commit()

    def get_best_params(self, prompt):
        # Query to get best params based on similar prompts
        pass
