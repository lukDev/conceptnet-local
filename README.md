# Local ConceptNet

ConceptNet and fastText embeddings with a local DB and API.
English only.

## Setup

Make sure the path to the DB is included in your environment variables:
```shell
CN_DB_PATH=<path>/cn.db
```

The DB file can be downloaded from here (~2.2GB):
https://drive.google.com/file/d/1ykKOa8oUzdiI5OShaQgyP8AZBH2iHDot/view?usp=sharing

Alternatively, the DB can be set up manually by navigating to the ```db_setup``` directory in this repository and following the instructions in the notebook there.

## Example Usage

Import:
```python
import conceptnet_local as cnl
```

### Concept Details

List of all concept IDs:
```python
all_concept_ids = cnl.get_all_concept_ids()
```

All edges connecting to a concept:
```python
edges = cnl.get_all_edges(cn_id="/c/en/example")
```

Relatedess between two concepts according to fastText embeddings:
```python
relatedness = cnl.get_relatedness("/c/en/example", "/c/en/test")
```

### Lowest-Cost Paths

Get a variant of A* by specifying cost and heuristic weights:
```python
from conceptnet_local import get_a_star_variant, CostFunction, HeuristicFunction

CustomAStar = get_a_star_variant(
    cost_weights={
        CostFunction.EDGE_COUNT: 2.,
        CostFunction.SIMILARITY_DIFFERENCE: 1.,
    },
    heuristic_weights={
        HeuristicFunction.SIMILARITY_TO_GOAL: 1.,
    }
)
```
or create a fully custom variant of A*:
```python
from conceptnet_local import AStar, Concept, Relation

class CustomAStar(AStar):
    def get_cost(self, source: Concept, target: Concept, relation: Relation, goal: Concept) -> float:
        ...

    def get_heuristic(self, current: Concept, goal: Concept) -> float:
        ...
```

Get the lowest-cost path:
```python
custom_a_star = CustomAStar()
path = custom_a_star.compute_path(input_concept="/c/en/example", output_concept="/c/en/test", print_time=True)

print(cnl.format_path(path=path))
```

### Custom Queries

Get a reference to the DB:
```python
connection, cursor = cnl.setup_sqlite_db()
```

Execute queries, e.g.:
```python
partial_id = "examp"
statement = cursor.execute("SELECT embedding FROM embeddings WHERE concept_id LIKE '%?%'", (partial_id,))
result = statement.fetchall()
```

Close DB connection:
```python
cnl.close_sqlite_db(db_connection=connection)
```

## Version History

### 1.0
- release 🎉

### 0.9
- concept extraction from text
- concept degree retrieval

### 0.8
- concepts table
- retrieval of similar concepts
- concept existence check

### 0.7
- separate utils file
- link label formatting
- natural-language path formatting

### 0.6
- no more FastText model built in
- embedding computation method optional in relatedness method
- optimized embedding retrieval from DB

### 0.5
- custom initialization method in A*

### 0.4
- FastText embeddings for arbitrary text

### 0.3
- retrieval of all concept IDs

### 0.2
- FastText embeddings

### 0.1
- reading links and embeddings from DB
- custom A* with configurable variants
- greedy version of Yen's algorithm
