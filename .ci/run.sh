#!/usr/bin/env bash

if [ "$1" = "2" ]; then
    echo "run python-pninexus tests"
    docker exec --user root ndts python setup.py test
    docker exec ndts python -m pytest test
else
    echo "run python3-pninexus tests"
    docker exec ndts python3 -m pytest test
fi
if [ "$?" != "0" ]; then exit 255; fi
