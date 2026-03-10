# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SQLite
DB_FILE = "data/parking_chatbot.db"

# Milvus Standalone
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "parking_knowledge"

# Parking info file
PARKING_INFO_FILE = "data/parking_info.txt"

# Text splitter
CHUNK_SIZE = 220
CHUNK_OVERLAP = 60

# Email configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")