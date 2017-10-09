# dket: Deep Knowledge Extraction from Text
The `DKET` project has the goal to devise a Neural Networks based Ontology Learning system that
doesn't rely on hand-crafted rules and it is trained in an end-to-end fashion. The motivation and
some preliminary investigation and results can be found in this [paper](https://link.springer.com/chapter/10.1007/978-3-319-49004-5_31).

## Installation
The best way to install and use the `DKET` package is to clone the git repository, and set up the proper virtual environment. Once cloned the repository, just move into the directoty and run the proper script to create and setp the proper Python 3 virtual environment.

    :~$ git clone git@github.com:dkmfbk/dket.git
    :~$ cd dket
    :~$ ./bin/dket-venv-setup gpu

or just run `./bin/dket-venv-setup` if you don't have a GPU card on your
machine. This will create a `.py3venv` directory and install all the dependecies
you need. Otherwise, You can install `DKET` as a regular pyhton package via
`pip`. Since `DKET` uses the [`LiTeFlow`](https://github.com/petrux/LiTeFlow)
library, you must resolve the link dependency during the instllation via the
`--process-dependency-links` directive:

    :~$ pip install --process-dependency-links https://github.com/dkmfbk/dket.git

But after installed `DKET` you need to install the proper `TensorFlow` version,
for GPU or CPU, by yourself.
