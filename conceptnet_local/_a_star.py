#########################################################
# heavily influenced by https://pypi.org/project/astar/ #
#########################################################
import time
from abc import ABC, abstractmethod
from math import inf
from sqlite3 import Cursor, Connection
from typing import Optional

from pydantic import BaseModel
from sortedcontainers import SortedList

from conceptnet_local._cn_service import setup_sqlite_db, close_sqlite_db, get_all_edges, Relation


class NoPathFoundError(Exception):
    pass


class Concept(BaseModel):
    id: str

    # cost of the cheapest path currently known from the input to this concept
    g_score: float = inf

    # f(c) = g(c) + h(c) | estimated cost of the cheapest path from input to output through this concept
    f_score: float = inf

    came_from: Optional["SearchRelation"] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "Concept"):
        return self.id == other.id

    def __lt__(self, other: "Concept") -> bool:
        return self.fscore < other.fscore


class SearchRelation(BaseModel):
    source_id: str
    target_id: str
    relation: Relation
    cost: float | None = None

    def __eq__(self, other: "SearchRelation"):
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relation == other.relation
        )

    def __hash__(self):
        return hash(self.source_id) ^ hash(self.target_id) ^ hash(self.relation)


Path = list[SearchRelation]


class ConceptDict(dict[str, Concept]):
    def __missing__(self, key: str) -> Concept:
        """Create and store a concept if it hasn't been created yet."""
        concept = Concept(id=key)
        self.__setitem__(key, concept)
        return concept


class PriorityQueue:
    def __init__(self):
        self.sorted_list = SortedList(key=lambda x: x.f_score)

    def push(self, item: Concept):
        self.sorted_list.add(item)

    def pop(self) -> Concept:
        item = self.sorted_list.pop(0)
        return item

    def remove(self, item: Concept):
        self.sorted_list.remove(item)

    def __len__(self) -> int:
        return len(self.sorted_list)

    def __contains__(self, item: Concept) -> bool:
        return item in self.sorted_list


class AStar(ABC):
    db_connection: Connection | None
    db_cursor: Cursor | None

    def compute_path(
        self, input_concept: str, output_concept: str, print_time: bool = False
    ) -> list[SearchRelation]:
        """
        Compute the shortest path from the given input to the given output.

        :param input_concept:   The concept from which the shortest path to the output should be computed.
        :param output_concept:  The concept to which the shortest path from the input should be computed.
        :param print_time:      A flag indicating whether the time it took the algorithm to run should be printed to stdout.
        :return:                The shortest path from input to output, given as a list of neighbor relations.
        """
        start_time = time.time()

        self.db_connection, self.db_cursor = setup_sqlite_db()

        concept_dict = ConceptDict()
        priority_queue = PriorityQueue()

        goal = concept_dict[output_concept]

        start = concept_dict[input_concept]
        start.g_score = 0
        start.f_score = self.get_heuristic(current=start, goal=goal)

        priority_queue.push(start)

        while len(priority_queue) > 0:
            current = priority_queue.pop()

            if current == goal:
                if print_time:
                    end_time = time.time()
                    print(
                        f"A* completed in {(end_time - start_time):.2f}s from {input_concept} to {output_concept}"
                    )

                self.close_db_connection()
                return _construct_path_backwards(
                    final_concept=current, concept_dict=concept_dict
                )

            search_relations = self.get_neighbors(concept=current)
            for search_relation in search_relations:
                neighbor = concept_dict[search_relation.target_id]
                cost = self.get_cost(
                    source=current,
                    target=neighbor,
                    relation=search_relation.relation,
                    goal=goal,
                )
                search_relation.cost = cost

                new_g_score = current.g_score + cost
                if new_g_score >= neighbor.g_score:
                    continue

                # remove the neighbor from the queue if it's already in there in order to trigger a re-sort when it's added again
                if neighbor in priority_queue:
                    priority_queue.remove(neighbor)

                neighbor.came_from = search_relation
                neighbor.g_score = new_g_score
                neighbor.f_score = new_g_score + self.get_heuristic(
                    current=neighbor, goal=goal
                )

                priority_queue.push(neighbor)

        self.close_db_connection()
        raise NoPathFoundError()

    def close_db_connection(self):
        close_sqlite_db(db_connection=self.db_connection)
        self.db_connection = None
        self.db_cursor = None

    @abstractmethod
    def get_cost(
        self, source: Concept, target: Concept, relation: Relation, goal: Concept
    ) -> float:
        """
        Compute and return the cost of going from the given source concept to the given target concept according to the given relation between them.

        :param source:      The source concept.
        :param target:      The target concept.
        :param relation:    The relation connecting the two concepts.
        :param goal:        The goal. May be used in the computation of the neighbor distance.
        :return:            The cost of going from the source concept to the target concept under the given relation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_heuristic(self, current: Concept, goal: Concept) -> float:
        """
        Compute and return the heuristic value for the cost to go from the given concept to the given goal.

        :param current: The concept from which the cost to go to the given goal should be estimated.
        :param goal:    The goal used for the heuristic cost computation.
        :return:        The heuristic value estimating the cost to go from the current concept to the goal.
        """
        raise NotImplementedError

    def get_neighbors(self, concept: Concept) -> list[SearchRelation]:
        """
        Retrieve all neighbors of the given concept.

        :param concept:         The concept for which all neighbors should be retrieved.
        :return:                A list containing all neighbors of the given concept, each along with the connecting relation.
        """
        edges: list[Relation] = get_all_edges(cn_id=concept.id, db_cursor=self.db_cursor)

        all_neighbours: set[SearchRelation] = set()
        for edge in edges:
            if edge.start == edge.end:
                continue

            neighbor_id = edge.end if edge.start == concept.id else edge.start

            relation = Relation(
                id=edge.id,
                start=edge.start,
                end=edge.end,
                rel=edge.rel,
                weight=edge.weight,
            )

            all_neighbours.add(
                SearchRelation(
                    source_id=concept.id, target_id=neighbor_id, relation=relation
                )
            )

        return list(all_neighbours)


def _construct_path_backwards(
    final_concept: Concept, concept_dict: ConceptDict
) -> list[SearchRelation]:
    """Construct the path to the goal by going backwards from the final concept."""
    path: list[SearchRelation] = []

    current = final_concept
    while current.came_from is not None:
        path.insert(0, current.came_from)
        current = concept_dict[current.came_from.source_id]

    return path


def format_path(path: Path) -> str:
    """Format the given path into a printable string."""
    lines: list[str] = []

    for sr in path:
        following_relation_direction = sr.source_id == sr.relation.start
        start_arrow = "<" if not following_relation_direction else ""
        end_arrow = ">" if following_relation_direction else ""

        relation_name = sr.relation.rel.replace("/r/", "")

        lines.append(
            f"{sr.source_id} {start_arrow}——{relation_name}——{end_arrow} {sr.target_id}"
        )

    return "\n".join(lines)
