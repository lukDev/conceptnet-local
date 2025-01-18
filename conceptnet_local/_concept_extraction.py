import itertools

import inflect
from conceptnet_local import _cn_service, _utils
from pydantic import BaseModel

_MAX_WORD_SEQUENCE_LENGTH = 5


class ConceptInText(BaseModel):
    concept_id: str
    text: str
    start_index: int  # including
    end_index: int  # excluding


#################
# Public Method #
#################

def get_concepts_in_text(
    text: str, ignore_grammatical_numbers: bool = False
) -> list[ConceptInText]:
    """
    Identify all concepts in the given text and return their IDs and position.

    :param text:                        The text in which to look for concepts.
    :param ignore_grammatical_numbers:  A flag indicating whether the grammatical numbers of the words in the given text should be ignored.
                                        If this flag is set, the singular/plural forms of the concepts in the text will be extracted as well, even if they were not originally there.
                                        Optional. Defaults to False.
    :return:                            A list containing all concepts found in the given text.
    """
    text_split, text_split_offsets = _split_text(text=text)

    concepts: list[ConceptInText] = []
    for word_count in range(1, _MAX_WORD_SEQUENCE_LENGTH + 1):
        concepts.extend(
            _get_concepts_with_word_count(
                word_count=word_count,
                text_split=text_split,
                text_split_offsets=text_split_offsets,
                ignore_grammatical_numbers=ignore_grammatical_numbers,
            )
        )

    concepts.sort(key=lambda cit: cit.start_index)
    return concepts


####################
# Helper Functions #
####################

_IGNORED_SYMBOLS = {"*", "-", ":", ".", ",", "?", "!", '"', "’", "(", ")", "[", "]", "{", "}"}
_INFLECT_ENGINE = inflect.engine()


def _split_text(text: str) -> tuple[list[str], list[int]]:
    """Split the given text into words and return the cleaned words and their offsets in the original text."""
    text = text.replace("\n", " ")
    text = text.replace("-", " ")
    text_split = text.split(" ")

    text_split_offsets = []
    current_offset = 0
    for i, part in enumerate(text_split):
        text_split_offsets.append(current_offset)
        current_offset += len(part) + 1  # part length plus one for the space character

    for i, part in enumerate(text_split):
        n_removed_chars = 0
        while len(part) > 0 and part[0] in _IGNORED_SYMBOLS:
            part = "" if len(part) == 1 else part[1:]
            n_removed_chars += 1
        text_split_offsets[i] += n_removed_chars

        while len(part) > 0 and part[-1] in _IGNORED_SYMBOLS:
            part = part[:-1]

        text_split[i] = part

    return text_split, text_split_offsets


def _get_word_variations(word: str) -> list[str]:
    """Get the singular/plural variations of the given word."""
    variations: set[str] = {word}
    variations.add(_INFLECT_ENGINE.plural(word))

    singular: str | False = _INFLECT_ENGINE.singular_noun(text=word)
    if singular is not False:
        variations.add(str(singular))

    return list(variations)


def _get_word_combinations(words: list[str]) -> list[tuple[str, ...]]:
    """Get all combinations of the singular/plural forms of the given words."""
    word_variations = [_get_word_variations(word=w) for w in words]
    return list(itertools.product(*word_variations))


def _get_concepts_with_word_count(
    word_count: int,
    text_split: list[str],
    text_split_offsets: list[int],
    ignore_grammatical_numbers: bool,
) -> list[ConceptInText]:
    """Retrieve all concepts from the given (split) text that have the given word count."""
    concepts: list[ConceptInText] = []

    for i in range(len(text_split)):
        if i + word_count > len(text_split):
            break

        words = text_split[i : i + word_count]

        if any(len(w) == 0 for w in words):
            continue

        word_combinations: list[tuple[str, ...]] = (
            _get_word_combinations(words=words) if ignore_grammatical_numbers else [tuple(words)]
        )
        for combination in word_combinations:
            concept_term = " ".join(combination)
            concept_term_sanitized = _utils.sanitize_term(term=concept_term)

            if _utils.is_word_stopword(word=concept_term_sanitized):
                continue

            cn_id = _utils.get_id_from_natural_concept(concept=concept_term_sanitized)

            links = _cn_service.get_all_edges(cn_id=cn_id)

            if len(links) == 0:
                continue

            offset = text_split_offsets[i]
            concepts.append(
                ConceptInText(
                    concept_id=cn_id,
                    text=concept_term,
                    start_index=offset,
                    end_index=offset + len(concept_term),
                )
            )

    return concepts
