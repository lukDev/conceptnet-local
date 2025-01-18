import re

from nltk.corpus import stopwords

from conceptnet_local._a_star import Path, SearchRelation

_CN_LANGUAGE_PREFIX = "/c/en/"
_STOPWORDS = stopwords.words("english")


##################
# Public Methods #
##################

def get_formatted_link_label(label: str) -> str:
    """
    Format the given edge label in a more readable way.

    :param label:   The label to format.
    :return:        The formatted label.
    """
    label = label.replace("/r/", "")  # assumed shape: /r/<relation>

    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", label)
    parts = [m.group(0) for m in matches]
    label = " ".join(parts)

    label = label.lower()
    return label


def format_path(path: Path, natural_language: bool = False) -> str:
    """
    Format the given path into a printable string.

    :param path:                The path to be formatted.
    :param natural_language:    A flag indicating whether the path should be formatting in a technical way (False) or into natural language (True).
                                Optional. Defaults to False.
    :return:                    The formatted path as a string.
    """
    formatting_method = _format_search_relation_natural if natural_language else _format_search_relation_technical
    lines: list[str] = [formatting_method(sr) for sr in path]
    return "\n".join(lines)


def get_concepts_from_path(path: Path) -> list[str]:
    """
    Retrieve the concepts from the given path and return them in order.

    :param path:    The path whose concepts should be retrieved.
    :return:        A list containing the IDs of the concepts on the given path, in order.
    """
    if len(path) == 0:
        return []

    return [sr.source_id for sr in path] + [path[-1].target_id]


def get_natural_concept_from_id(concept_id: str) -> str:
    """
    Convert the given concept ID to its natural form.

    :param concept_id:  The concept ID to convert to its natural form.
    :return:            The natural form of the given concept ID.
    """
    return concept_id.replace(_CN_LANGUAGE_PREFIX, "").replace("_", " ")


def get_id_from_natural_concept(concept: str) -> str:
    """
    Convert the given natural concept to its ID.

    :param concept: The natural concept to convert to its ID.
    :return:        The ID of the given natural concept.
    """
    return f"{_CN_LANGUAGE_PREFIX}{concept.lower().replace(' ', '_')}"


def sanitize_term(term: str) -> str:
    """
    Sanitize the given term for conversion to a concept ID.

    :param term:    The term to be sanitized.
    :return:        The sanitized term.
    """
    term = term.lower()

    multiple_whitespace_regex = re.compile(" {2,}")
    term = multiple_whitespace_regex.sub(" ", term)

    return term


######################
# Internal Functions #
######################


def is_word_stopword(word: str) -> bool:
    """
    Check whether the given word is a stopword.

    :param word:    The word to be checked.
    :return:        A flag indicating whether the given word is a stopword.
    """
    return word in _STOPWORDS


####################
# Helper Functions #
####################


def _format_search_relation_natural(sr: SearchRelation) -> str:
    """Format the given search relation in a natural way."""
    start_concept = get_natural_concept_from_id(concept_id=sr.relation.start)
    end_concept = get_natural_concept_from_id(concept_id=sr.relation.end)

    relation_name = get_formatted_link_label(label=sr.relation.rel)

    return f"{start_concept} {relation_name} {end_concept}"


def _format_search_relation_technical(sr: SearchRelation) -> str:
    """Format the given search relation in a technical way."""
    start_arrow: str
    end_arrow: str
    relation_name: str

    if sr.relation is None:
        start_arrow = ""
        end_arrow = ">"
        relation_name = ""
    else:
        following_relation_direction = sr.source_id == sr.relation.start
        start_arrow = "<" if not following_relation_direction else ""
        end_arrow = ">" if following_relation_direction else ""

        relation_name = sr.relation.rel.replace("/r/", "")

    return f"{sr.source_id} {start_arrow}——{relation_name}——{end_arrow} {sr.target_id}"