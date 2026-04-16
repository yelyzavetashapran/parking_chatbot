import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from reservation_graph import build_reservation_graph

CHECKPOINT_DB = "data/checkpoints.db"

_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA busy_timeout=5000")

checkpointer = SqliteSaver(_conn)
reservation_graph = build_reservation_graph(checkpointer)
