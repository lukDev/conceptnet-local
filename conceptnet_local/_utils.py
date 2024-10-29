import re

from conceptnet_local._a_star import Path, SearchRelation


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


####################
# Helper Functions #
####################


def _get_concept_from_cn_id(cn_id: str) -> str:
    """Extract the concept name from the given CN ID."""
    return cn_id.replace("/c/en/", "").replace("_", " ")


def _format_search_relation_natural(sr: SearchRelation) -> str:
    """Format the given search relation in a natural way."""
    start_concept = _get_concept_from_cn_id(cn_id=sr.relation.start)
    end_concept = _get_concept_from_cn_id(cn_id=sr.relation.end)

    relation_name = get_formatted_link_label(label=sr.relation.rel)

    return f"{start_concept} {relation_name} {end_concept}"


def _format_search_relation_technical(sr: SearchRelation) -> str:
    """Format the given search relation in a technical way."""
    following_relation_direction = sr.source_id == sr.relation.start
    start_arrow = "<" if not following_relation_direction else ""
    end_arrow = ">" if following_relation_direction else ""

    relation_name = sr.relation.rel.replace("/r/", "")

    return f"{sr.source_id} {start_arrow}——{relation_name}——{end_arrow} {sr.target_id}"