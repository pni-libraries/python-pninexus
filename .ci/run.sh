#!/usr/bin/env bash

if [ "$1" = "2" ]; then
    echo "run python-pninexus tests"
    docker exec ndts bash -c "ls -ltr src"
    docker exec --user root ndts python setup.py test
else
    echo "run python3-pninexus tests"
    docker exec ndts python3 -m pytest test
fi
if [ "$?" != "0" ]; then exit 255; fi
