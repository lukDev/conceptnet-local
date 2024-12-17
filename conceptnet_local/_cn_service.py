import io
import os
import sqlite3
from functools import wraps
from sqlite3 import Connection, Cursor
from typing import Callable

import numpy as np
from pydantic import BaseModel


class Relation(BaseModel):
    id: str
    start: str
    end: str
    rel: str
    weight: float

    def __eq__(self, other: "Relation"):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


EmbeddingComputationMethod = Callable[[str], np.ndarray]


##################
# Public Methods #
##################


def get_all_edges(cn_id: str, db_cursor: Cursor | None = None) -> list[Relation]:
    """
    Retrieve all edges connected to the concept with the given ID.

    :param cn_id:       The ID of the concept for which all edges should be retrieved.
    :param db_cursor:   The DB cursor to use in the queries (optional).
    :return:            A list containing all edges connected to the given concept.
    """
    return _db_get_relations(cn_id=cn_id, db_cursor=db_cursor)


def get_relatedness(
    cn_id_1: str,
    cn_id_2: str,
    compute_method: EmbeddingComputationMethod | None = None,
    db_cursor: Cursor | None = None,
) -> float:
    """
    Compute and return the relatedness of the concepts with the given CN IDs.

    :param cn_id_1:         The ID of the first concepts for which the relatedness with the second concept should be computed.
    :param cn_id_2:         The ID of the second concepts for which the relatedness with the first concept should be computed.
    :param compute_method:  The method to use for computing the embeddings (optional).
                            If this is not given, the DB will be used to retrieve the embeddings.
    :param db_cursor:       The DB cursor to use in the queries (optional).
    :return:                The relatedness of the given concepts, as a float in [-1, 1].
    """
    try:
        if compute_method is not None:
            e1 = compute_method(_get_concept_from_cn_id(cn_id=cn_id_1))
            e2 = compute_method(_get_concept_from_cn_id(cn_id=cn_id_2))
        else:
            e1 = _db_get_embedding(cn_id=cn_id_1, db_cursor=db_cursor)
            e2 = _db_get_embedding(cn_id=cn_id_2, db_cursor=db_cursor)
    except Exception:
        return -1

    cosine_similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    return cosine_similarity


def get_all_concept_ids(db_cursor: Cursor | None = None) -> list[str]:
    """
    Retrieve the IDs of all concepts within CN.

    :param db_cursor:   The DB cursor to use in the query (optional).
    :return:            A list containing the IDs of all concepts.
    """
    return _db_get_all_concepts(db_cursor=db_cursor)


def get_similar_concepts(search_term: str, n_concepts: int, db_cursor: Cursor | None = None) -> list[str]:
    """
    Retrieve those concepts from the DB that are similar to the given search term.

    :param search_term: The search term for which similar concepts should be retrieved.
    :param n_concepts:  The number of similar concepts that should be retrieved.
    :param db_cursor:   The DB cursor to use in the query (optional).
    :return:            A list containing the IDs of the similar concepts.
    """
    search_term = search_term.replace(" ", "_")
    search_term = f"%{search_term}%"

    results = _db_get_similar_concepts(search_term=search_term, n_concepts=n_concepts, db_cursor=db_cursor)
    return results


############
# DB Setup #
############


def setup_sqlite_db() -> tuple[Connection, Cursor]:
    """Set up the connection to the SQLite database containing the CN data."""
    cn_db_path = os.getenv("CN_DB_PATH")

    if cn_db_path is None:
        raise ValueError("CN DB path is not specified in the environment variables")

    if not os.path.isfile(cn_db_path):
        raise ValueError("CN DB path does not point to a file")

    sqlite3.register_adapter(np.ndarray, _db_adapt_array)
    sqlite3.register_converter("ARRAY", _db_convert_array)

    db_connection = sqlite3.connect(database=cn_db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    db_cursor = db_connection.cursor()

    return db_connection, db_cursor


def close_sqlite_db(db_connection: Connection):
    """Close the connection to the SQLite database containing the CN data."""
    db_connection.commit()
    db_connection.close()


def with_cn_db():
    """Return a decorator for using the CN database."""

    def decorator(main_function):
        @wraps(main_function)
        def wrapper(*args, **kwargs):
            if "db_cursor" in kwargs and kwargs["db_cursor"] is not None:
                return main_function(*args, **kwargs)

            db_connection, db_cursor = setup_sqlite_db()

            kwargs["db_cursor"] = db_cursor
            result = main_function(*args, **kwargs)

            close_sqlite_db(db_connection=db_connection)

            return result

        return wrapper

    return decorator


##############
# DB Queries #
##############


# TODO: make more flexible, i.e., add filters like in the public ConceptNet API
@with_cn_db()
def _db_get_relations(cn_id: str, db_cursor: Cursor) -> list[Relation]:
    """Retrieve all relations (edges) connected to the concept with the given ID."""
    statement = db_cursor.execute(
        "SELECT * FROM relations WHERE (start = ? OR end = ?) AND rel != '/r/ExternalURL'",
        (cn_id, cn_id),
    )
    result = statement.fetchall()

    relations = [Relation(id=r[0], start=r[1], end=r[2], rel=r[3], weight=r[4]) for r in result]
    return relations


@with_cn_db()
def _db_get_embedding(cn_id: str, db_cursor: Cursor) -> np.ndarray:
    """Retrieve the embedding for the concept with the given ID from the DB."""
    statement = db_cursor.execute("SELECT embedding FROM embeddings WHERE concept_id = ?", (cn_id,))
    result = statement.fetchone()

    if result is None:
        raise ValueError(f"no concept with ID {cn_id}")

    embedding = result[0]
    return embedding


@with_cn_db()
def _db_get_all_concepts(db_cursor: Cursor) -> list[str]:
    """Retrieve the IDs of all concepts in the DB."""
    statement = db_cursor.execute(
        "SELECT concept_id FROM embeddings",
    )
    result = statement.fetchall()

    return [c[0] for c in result]


@with_cn_db()
def _db_get_similar_concepts(search_term: str, n_concepts: int, db_cursor: Cursor) -> list[str]:
    """Retrieve the IDs of similar concepts to the given search terms."""
    statement = db_cursor.execute(
        "SELECT concept_id FROM embeddings WHERE concept_id LIKE ? LIMIT ?", (search_term, n_concepts),
    )
    result = statement.fetchall()

    return [c[0] for c in result]


####################
# Helper Functions #
####################


def _get_concept_from_cn_id(cn_id: str) -> str:
    """Extract the concept name from the given CN ID."""
    return cn_id.replace("/c/en/", "").replace("_", " ")


def _db_adapt_array(array: np.ndarray) -> sqlite3.Binary:
    """Convert the given numpy array to a blob for the DB."""
    out = io.BytesIO()
    np.save(file=out, arr=array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _db_convert_array(text: bytes) -> np.ndarray:
    """Convert the given DB blob to a numpy array."""
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(file=out)
