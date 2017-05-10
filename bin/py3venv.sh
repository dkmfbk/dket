# !/bin/bash

# if .py3venv doesn't exists, create it,
# otherwise activate the virtual environment
if [ -d .py3venv ]; then
    source .py3venv/bin/activate
else
    virtualenv .py3venv --python=python3
    source .py3venv/bin/activate
    pip install -r venvrequirements.txt        
fi