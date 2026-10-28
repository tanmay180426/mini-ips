"""Storage package for Mini-IDS.

Manages persistent storage of normalized packets and security alerts
in an SQLite database.
"""

from storage.database import DEFAULT_DB_PATH, DatabaseManager, init_db

__all__ = [
    "DatabaseManager",
    "init_db",
    "DEFAULT_DB_PATH",
]
