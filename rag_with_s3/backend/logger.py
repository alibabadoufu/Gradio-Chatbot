import sqlite3
import datetime
import uuid
import pandas as pd

DATABASE_FILE = "usage_logs.db"

def init_db():
    """Initializes the SQLite database and creates the conversations table."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            selected_document TEXT,
            conversation_id TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_conversation(user_message: str, ai_response: str, selected_document: str = None, conversation_id: str = None):
    """Logs a conversation turn to the database."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    message_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (id, timestamp, user_message, ai_response, selected_document, conversation_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (message_id, timestamp, user_message, ai_response, selected_document, conversation_id))
    conn.commit()
    conn.close()
    return conversation_id

def get_usage_data(start_date: str = None, end_date: str = None):
    """Retrieves conversation logs from the database within a date range."""
    conn = sqlite3.connect(DATABASE_FILE)
    query = "SELECT * FROM conversations"
    params = []

    if start_date and end_date:
        query += " WHERE date(timestamp) BETWEEN ? AND ?"
        params = [start_date, end_date]
    elif start_date:
        query += " WHERE date(timestamp) >= ?"
        params = [start_date]
    elif end_date:
        query += " WHERE date(timestamp) <= ?"
        params = [end_date]
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_daily_trends(start_date: str = None, end_date: str = None):
    """Calculates daily conversation trends."""
    df = get_usage_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["date", "conversations"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_counts = df.groupby("date")["conversation_id"].nunique().reset_index(name="conversations")
    return daily_counts

def get_conversation_dataframe(start_date: str = None, end_date: str = None):
    """Returns a DataFrame of detailed conversation logs."""
    df = get_usage_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "conversation_id", "user_message", "ai_response", "selected_document"])
    return df[["timestamp", "conversation_id", "user_message", "ai_response", "selected_document"]]

def get_key_metrics(start_date: str = None, end_date: str = None):
    """Calculates key usage metrics."""
    df = get_usage_data(start_date, end_date)
    total_conversations = df["conversation_id"].nunique() if not df.empty else 0
    total_messages = len(df) if not df.empty else 0
    unique_documents_chatted = df["selected_document"].nunique() if not df.empty else 0
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "unique_documents_chatted": unique_documents_chatted,
    }

# Initialize the database when the module is imported
init_db()

