# !/bin/bash

# if .py2venv doesn't exists, create it,
# otherwise activate the virtual environment
if [ -d .py2venv ]; then
    source .py2venv/bin/activate
else
    virtualenv .py2venv --python=python2
    source .py2venv/bin/activate
    pip install -r venvrequirements.txt        
fi