# dket: Deep Knowledge Extraction from Text
The `DKET` project has the goal to devise a Neural Networks based Ontology Learning system that doesn't rely on hand-crafted rules and it is trained in an end-to-end fashion. The motivation and some preliminary investigation and results can be found in this [paper](https://link.springer.com/chapter/10.1007/978-3-319-49004-5_31).

## Installation
You can install `DKET` as a regular pyhton package via `pip`. Since `DKET` uses the [`LiTeFlow`](https://github.com/petrux/LiTeFlow) library, you must resolve the link dependency during the instllation via the `--process-dependency-links` directive:

    :~$ pip install --process-dependency-links https://github.com/petrux/dket.git