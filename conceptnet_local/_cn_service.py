import os
import sqlite3
from functools import wraps
from sqlite3 import Connection, Cursor

import fasttext
import numpy as np
from pydantic import BaseModel

_FASTTEXT_PATH = os.getenv("CN_FASTTEXT_PATH")
if _FASTTEXT_PATH is None:
    raise ValueError("FastText model path is not specified in the environment variables")
_FASTTEXT_MODEL = fasttext.load_model(_FASTTEXT_PATH)


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


def get_relatedness(cn_id_1: str, cn_id_2: str, db_cursor: Cursor | None = None) -> float:
    """
    Compute and return the relatedness of the concepts with the given CN IDs.

    :param cn_id_1:     The ID of the first concepts for which the relatedness with the second concept should be computed.
    :param cn_id_2:     The ID of the second concepts for which the relatedness with the first concept should be computed.
    :param db_cursor:   The DB cursor to use in the queries (optional).
    :return:            The relatedness of the given concepts, as a float in [-1, 1].
    """
    e1 = _FASTTEXT_MODEL.get_sentence_vector(text=_get_concept_from_cn_id(cn_id=cn_id_1))
    e2 = _FASTTEXT_MODEL.get_sentence_vector(text=_get_concept_from_cn_id(cn_id=cn_id_2))

    # TODO: figure out why using the DB results in much slower A* runs
    # try:
    #     e1 = _db_get_embedding(cn_id=cn_id_1, db_cursor=db_cursor)
    #     e2 = _db_get_embedding(cn_id=cn_id_2, db_cursor=db_cursor)
    # except ValueError:
    #     return -1

    cosine_similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    return cosine_similarity


def get_all_concept_ids(db_cursor: Cursor | None = None) -> list[str]:
    """
    Retrieve the IDs of all concepts within CN.

    :param db_cursor:   The DB cursor to use in the query (optional).
    :return:            A list containing the IDs of all concepts.
    """
    return _db_get_all_concepts(db_cursor=db_cursor)


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

    db_connection = sqlite3.connect(database=cn_db_path)
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

    relations = [
        Relation(id=r[0], start=r[1], end=r[2], rel=r[3], weight=r[4])
        for r in result
    ]
    return relations


@with_cn_db()
def _db_get_embedding(cn_id: str, db_cursor: Cursor) -> np.ndarray:
    """Retrieve the embedding for the concept with the given ID from the DB."""
    statement = db_cursor.execute(
        "SELECT embedding FROM embeddings WHERE concept_id = ?", (cn_id,)
    )
    result = statement.fetchone()

    if result is None:
        raise ValueError(f"no concept with ID {cn_id}")

    embedding_string = result[0]
    embedding = np.fromstring(embedding_string, sep=" ")
    return embedding


@with_cn_db()
def _db_get_all_concepts(db_cursor: Cursor) -> list[str]:
    """Retrieve the embedding for the concept with the given ID from the DB."""
    statement = db_cursor.execute(
        "SELECT concept_id FROM embeddings",
    )
    result = statement.fetchall()

    return [c[0] for c in result]


####################
# Helper Functions #
####################


def _get_concept_from_cn_id(cn_id: str) -> str:
    """Extract the concept name from the given CN ID."""
    concept = cn_id.replace("/c/en/", "")
    concept = concept.replace("_", " ")
    return concept
