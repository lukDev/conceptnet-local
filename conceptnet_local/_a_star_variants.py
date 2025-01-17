import enum
from sqlite3 import Cursor
from typing import Callable, Type

import numpy as np
from conceptnet_local._cn_service import get_relatedness, EmbeddingComputationMethod
from conceptnet_local._a_star import Concept, Relation, AStar
from pydantic import NonNegativeFloat


# ====================
# === cost methods ===
# ====================


def _get_cost_edge_count(
    source: Concept,
    target: Concept,
    relation: Relation,
    goal: Concept,
    db_cursor: Cursor | None,
    embedding_method: EmbeddingComputationMethod | None,
) -> float:
    """Compute the edge-count cost of going from the source concept to the target concept (which is always 1)."""
    return 1.0


def _get_cost_edge_weight_natural(
    source: Concept,
    target: Concept,
    relation: Relation,
    goal: Concept,
    db_cursor: Cursor | None,
    embedding_method: EmbeddingComputationMethod | None,
) -> float:
    """Compute the natural edge-weight cost of going from the source concept to the target concept (which is the inverse CN edge weight)."""
    return (
        1.0 / relation.weight
    )  # high weight <=> low cost, since weight measures plausibility of the relation


def _get_cost_edge_weight_inverse(
    source: Concept,
    target: Concept,
    relation: Relation,
    goal: Concept,
    db_cursor: Cursor | None,
    embedding_method: EmbeddingComputationMethod | None,
) -> float:
    """Compute the inverse edge-weight cost of going from the source concept to the target concept (which is the CN edge weight)."""
    return relation.weight  # more plausibility <=> higher cost, to find more obscure paths


def _get_cost_similarity_difference(
    source: Concept,
    target: Concept,
    relation: Relation,
    goal: Concept,
    db_cursor: Cursor | None,
    embedding_method: EmbeddingComputationMethod | None,
) -> float:
    """Compute the similarity-difference cost of going from the source concept to the target concept."""
    similarity_source = get_relatedness(
        cn_id_1=source.id, cn_id_2=goal.id, compute_method=embedding_method, db_cursor=db_cursor
    )  # in [-1, 1], 1 being close
    similarity_target = get_relatedness(
        cn_id_1=target.id, cn_id_2=goal.id, compute_method=embedding_method, db_cursor=db_cursor
    )  # in [-1, 1], 1 being close

    similarity_difference = (
        similarity_target - similarity_source
    )  # in [-2, 2], 2 being the largest possible improvement
    return 1.0 - similarity_difference / 2  # in [0, 2], 0 being the largest possible improvement


# =========================
# === heuristic methods ===
# =========================


def _get_heuristic_similarity(
    current: Concept,
    goal: Concept,
    db_cursor: Cursor | None,
    embedding_method: EmbeddingComputationMethod | None,
) -> float:
    """Compute the similarity between the given current concept and the goal."""
    cosine_similarity = get_relatedness(
        cn_id_1=current.id, cn_id_2=goal.id, compute_method=embedding_method, db_cursor=db_cursor
    )  # in [-1, 1], 1 being close
    return 1 - cosine_similarity  # in [0, 2], 0 being close


# =========================
# === model definitions ===
# =========================


class CostFunction(str, enum.Enum):
    EDGE_COUNT = "edge_count"  # every edge has equal cost
    EDGE_WEIGHT_NATURAL = "edge_weight_natural"  # relation plausible <=> low cost, i.e. inverse of CN weights
    EDGE_WEIGHT_INVERSE = "edge_weight_inverse"  # relation plausible <=> high cost, i.e. CN weights directly
    SIMILARITY_DIFFERENCE = (
        "similarity_difference"  # increase in similarity with the goal from source to target <=> low cost
    )


CostWeightMap = dict[CostFunction, NonNegativeFloat]
CostFunctionMap = dict[CostFunction, Callable]

COST_FUNCTIONS: CostFunctionMap = {
    CostFunction.EDGE_COUNT: _get_cost_edge_count,
    CostFunction.EDGE_WEIGHT_NATURAL: _get_cost_edge_weight_natural,
    CostFunction.EDGE_WEIGHT_INVERSE: _get_cost_edge_weight_inverse,
    CostFunction.SIMILARITY_DIFFERENCE: _get_cost_similarity_difference,
}


class HeuristicFunction(str, enum.Enum):
    SIMILARITY_TO_GOAL = "similarity_to_goal"  # high similarity to goal <=> low heuristic value


HeuristicWeightMap = dict[HeuristicFunction, NonNegativeFloat]
HeuristicFunctionMap = dict[HeuristicFunction, Callable]

HEURISTIC_FUNCTIONS: HeuristicFunctionMap = {
    HeuristicFunction.SIMILARITY_TO_GOAL: _get_heuristic_similarity
}


# ==============================
# === algorithm construction ===
# ==============================


def _get_weighted_values(
    weight_map: CostWeightMap | HeuristicWeightMap,
    function_map: CostFunctionMap | HeuristicFunctionMap,
    arguments: dict,
) -> float:
    """
    Compute the weighted average of the results of the given function types.

    :param weight_map:      A dictionary mapping from a function type to the associated weight.
    :param function_map:    A dictionary mapping from a function type (enum) to the actual corresponding function.
    :param arguments:       The names arguments to be used when calling the evaluation functions.
    :return:                The weighted average of the results of the given function types.
    """
    if len(weight_map) == 0:
        return 0.0

    weights: np.ndarray = np.array([w for w in weight_map.values()])
    values: list[float] = []
    for i, function_type in enumerate(weight_map.keys()):
        if weights[i] == 0:
            values.append(0.)
            continue

        function = function_map[function_type]
        values.append(function(**arguments))
    values: np.ndarray = np.array(values)

    weighted_values = weights * values
    return weighted_values.sum() / weights.sum()


def get_a_star_variant(
    cost_weights: CostWeightMap,
    heuristic_weights: HeuristicWeightMap,
    embedding_method: EmbeddingComputationMethod | None = None,
) -> Type[AStar]:
    """
    Get a variant of A* based on the given parameters.

    :param cost_weights:        The weights of the different cost function to be used within the constructed A* instance.
                                At least one of these weights must be > 0.
    :param heuristic_weights:   The weights of the different heuristic functions to be used within the constructed A* instance.
    :param embedding_method:    The method to use for computing embeddings (optional).
                                If this is not given, the DB will be used to retrieve any embeddings that may be needed.
    :return:                    The A* class specified by the given parameters.
    """
    if len(cost_weights) == 0:
        raise ValueError("At least one weight must be given.")

    all_cost_weights = [v for v in cost_weights.values()]
    if sum(all_cost_weights) <= 0.0:
        raise ValueError("At least one weight must be > 0.")

    class CustomAStar(AStar):
        def get_cost(
            self,
            source: Concept,
            target: Concept,
            relation: Relation,
            goal: Concept,
        ) -> float:
            function_arguments = {
                "source": source,
                "target": target,
                "relation": relation,
                "goal": goal,
                "db_cursor": self.db_cursor,
                "embedding_method": embedding_method,
            }
            return _get_weighted_values(
                weight_map=cost_weights,
                function_map=COST_FUNCTIONS,
                arguments=function_arguments,
            )

        def get_heuristic(self, current: Concept, goal: Concept) -> float:
            function_arguments = {
                "current": current,
                "goal": goal,
                "db_cursor": self.db_cursor,
                "embedding_method": embedding_method,
            }
            return _get_weighted_values(
                weight_map=heuristic_weights,
                function_map=HEURISTIC_FUNCTIONS,
                arguments=function_arguments,
            )

    return CustomAStar
