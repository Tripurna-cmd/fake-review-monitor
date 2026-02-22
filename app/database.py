import sqlite3
import os
from datetime import datetime

DB_PATH = 'data/reviews_history.db'

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text   TEXT NOT NULL,
            prediction    TEXT NOT NULL,
            confidence    REAL NOT NULL,
            fake_prob     REAL NOT NULL,
            real_prob     REAL NOT NULL,
            analyzed_at   TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_review(review_text, prediction, confidence, fake_prob, real_prob):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO review_history 
        (review_text, prediction, confidence, fake_prob, real_prob, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        review_text,
        prediction,
        confidence,
        fake_prob,
        real_prob,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

def get_all_reviews():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, review_text, prediction, confidence, 
               fake_prob, real_prob, analyzed_at 
        FROM review_history 
        ORDER BY analyzed_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM review_history")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM review_history WHERE prediction='FAKE'")
    fake_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM review_history WHERE prediction='REAL'")
    real_count = cursor.fetchone()[0]
    conn.close()
    return {'total': total, 'fake': fake_count, 'real': real_count}