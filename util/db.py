import os
import os.path
import sqlite3
import typing as t

from util.const import DATA_DIR

def _get_sqlite3_thread_safety():

    # Mape value from SQLite's THREADSAFE to Python's DBAPI 2.0
    # threadsafety attribute.
    sqlite_threadsafe2python_dbapi = {0: 0, 2: 1, 1: 3}
    with sqlite3.connect(":memory:") as tx:
        threadsafety = tx.execute("""
        SELECT * FROM pragma_compile_options
        WHERE compile_options LIKE 'THREADSAFE=%'
        """).fetchone()[0]

    threadsafety_value = int(threadsafety.split("=")[1])

    return sqlite_threadsafe2python_dbapi[threadsafety_value]

_sqlite_connections: t.Dict[str, sqlite3.Connection] = {}
def sqlite_get(name: str = 'main') -> sqlite3.Connection:
    global _sqlite_connections
    if not name in _sqlite_connections:
        _sqlite_connections[name] = sqlite3.connect(
            DATA_DIR / f'{name}.sqlite3',
            check_same_thread=_get_sqlite3_thread_safety() != 3,
            timeout=120,
        )
    return _sqlite_connections[name]

def sqlite_destroy(name: str) -> None:
    if os.path.exists(DATA_DIR / f'{name}.sqlite3'):
        os.remove(DATA_DIR / f'{name}.sqlite3')
