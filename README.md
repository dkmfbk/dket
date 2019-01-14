# dket: Deep Knowledge Extraction from Text
The `DKET` project has the goal to devise a Neural Networks based Ontology Learning system that
doesn't rely on hand-crafted rules and it is trained in an end-to-end fashion. The approach, the datasets, and the experimental evaluation are described in this [paper](https://www.sciencedirect.com/science/article/pii/S1570826818300507).

## Installation
The best way to install and use the `DKET` package is to clone the git repository, and set up the proper virtual environment. Once cloned the repository, just move into the directory and run the proper script to create and setup the proper Python 3 virtual environment.

    :~$ git clone git@github.com:dkmfbk/dket.git
    :~$ cd dket
    :~$ ./bin/dket-venv-setup gpu

or just run `./bin/dket-venv-setup` if you don't have a GPU card on your
machine. This will create a `.py3venv` directory and install all the dependencies
you need. Otherwise, You can install `DKET` as a regular Pyhton package via
`pip`. Since `DKET` uses the [`LiTeFlow`](https://github.com/petrux/LiTeFlow)
library, you must resolve the link dependency during the installation via the
`--process-dependency-links` directive:

    :~$ pip install --process-dependency-links https://github.com/dkmfbk/dket.git

But after installed `DKET` you need to install the proper `TensorFlow` version,
for GPU or CPU, by yourself.

## Run the experiments
All the experimental settings are stored as `.json` files in the `experiments`
folder. To run them, after activating the `.py3venv` virtual environment, just run:

    :~$ ./bin/dket-experiment-run --config experiments/<EXP>.json

where `<EXP>` is the name of the experimental setting that you want to use. The full options for the `./bin/dket-experiment-run` are available at:

    :~$ ./bin/dket-experiment-run --help
    
## Resources
The folder `resources` contains:
- the manually curated dataset: 500 sentence-formula pairs, 75 used for training (`reference_set.train.tsv`) and 425 used for testing (`reference_set.test.tsv`);
- the grammar (`grammar.cfg`) used for generating the synthetic datasets.
