import sqlite3
from functools import wraps
from sqlite3 import Connection, Cursor

import numpy as np
from pydantic import BaseModel


class CnDbRelation(BaseModel):
    id: str
    start: str
    end: str
    rel: str
    weight: float


##################
# Public Methods #
##################


def get_all_edges(cn_id: str, db_cursor: Cursor | None = None) -> list[CnDbRelation]:
    """
    Retrieve all edges connected to the concept with the given ID.

    :param cn_id:       The ID of the concept for which all edges should be retrieved.
    :param db_cursor:   The DB cursor to use in the queries (optional).
    :return:            A list containing all edges connected to the given concept.
    """
    return _db_get_relations(cn_id=cn_id, db_cursor=db_cursor)


def get_relatedness(*cn_ids: str, db_cursor: Cursor | None = None) -> float:
    """
    Compute and return the relatedness of the concepts with the given CN IDs.

    :param cn_ids:      The IDs of the concepts for which the relatedness should be computed.
    :param db_cursor:   The DB cursor to use in the queries (optional).
    :return:            The relatedness of the given concepts, as a float in [-1, 1].
    """
    if len(cn_ids) != 2:
        raise ValueError(
            f"relatedness can only be computed between 2 concepts, but {len(cn_ids)} were given"
        )

    try:
        embeddings = [_db_get_embedding(cn_id=cn_id, db_cursor=db_cursor) for cn_id in cn_ids]
    except ValueError:
        return -1

    e1, e2 = embeddings
    cosine_similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    return cosine_similarity


############
# DB Setup #
############


def setup_sqlite_db() -> tuple[Connection, Cursor]:
    """Set up the connection to the SQLite database containing the CN data."""
    db_connection = sqlite3.connect(database="../data/cn.db")
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
def _db_get_relations(cn_id: str, db_cursor: Cursor) -> list[CnDbRelation]:
    """Retrieve all relations (edges) connected to the concept with the given ID."""
    statement = db_cursor.execute(
        "SELECT * FROM relations WHERE (start = ? OR end = ?) AND rel != '/r/ExternalURL'",
        (cn_id, cn_id),
    )
    result = statement.fetchall()

    relations = [
        CnDbRelation(id=r[0], start=r[1], end=r[2], rel=r[3], weight=r[4])
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
