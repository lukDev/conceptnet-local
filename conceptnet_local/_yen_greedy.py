import random
import time
from typing import Type

from pydantic import BaseModel

import _a_star


class _PathWithHash(BaseModel):
    path: _a_star.Path

    def __hash__(self):
        return hash(tuple(self.path))


def get_offshoot_paths(
    custom_a_star: Type[_a_star.AStar],
    original_path: _a_star.Path,
    max_paths: int | None = None,
    print_time: bool = False,
) -> list[_a_star.Path]:
    """
    Use a greedy version of Yen's algorithm to compute offshoot paths of the given path.
    An offshoot path will start with the same input concept and end with the same output concept as the original path.

    Limit the computation to at most the given maximum.

    :param custom_a_star:   The variant of A* that should be used to compute the offshoot paths.
    :param original_path:   The path off of which the offshoots paths should be computed.
                            This will typically be the shortest path.
                            This must have been created using the same cost metric as the given A* variant.
    :param max_paths:       The maximum number of paths to be computed.
                            If the given path has more nodes than this maximum, not all of these nodes will be branched off from.
                            If None, no limit will be imposed on the number of paths.
                            Default: None.
    :param print_time:      A flag indicating whether the time it took the algorithm to run should be printed to stdout.
    :return:                A list containing at most the given maximum number of offshoot paths off of the given path.
    """
    start_time = time.time()

    paths: set[_PathWithHash] = set()

    path_nodes_without_goal = [r.source_id for r in original_path]
    goal = original_path[-1].target_id

    indices_to_branch = [i for i in range(len(path_nodes_without_goal))]
    if max_paths is not None and len(path_nodes_without_goal) > max_paths:
        indices_to_branch = random.sample(population=indices_to_branch, k=max_paths)

    for i in indices_to_branch:
        spur_node = path_nodes_without_goal[i]
        root_path = original_path[:i]

        blocked_relations: set[_a_star.Relation] = {original_path[i].relation}
        for root_sr in root_path:
            blocked_relations.add(root_sr.relation)

        class BlockedAStar(custom_a_star):
            def get_neighbors(
                self, concept: _a_star.Concept
            ) -> list[_a_star.SearchRelation]:
                all_neighbors = super().get_neighbors(concept=concept)
                return [
                    neighbor_sr
                    for neighbor_sr in all_neighbors
                    if neighbor_sr.relation not in blocked_relations
                ]

        blocked_a_star = BlockedAStar()

        spur_path = blocked_a_star.compute_path(
            input_concept=spur_node, output_concept=goal
        )
        total_path = root_path + spur_path
        paths.add(_PathWithHash(path=total_path))

    if print_time:
        end_time = time.time()
        print(
            f"Yen-greedy completed in {(end_time - start_time):.2f}s and found {len(paths)} new paths"
        )

    return [p.path for p in paths]
