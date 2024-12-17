# Local ConceptNet

ConceptNet and CN Numberbatch with a local DB and API.

The DB can be set up by navigating to the ```db_setup``` directory and following the instructions in the notebook there.

## Version History

### 0.8
- concepts table
- retrieval of similar concepts

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
