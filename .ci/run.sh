#!/usr/bin/env bash

if [ "$1" = "2" ]; then
    echo "run python-pninexus"
    docker exec --user root ndts python setup.py test
    # docker exec -it ndts python setup.py test
else
    echo "run python3-pninexus"
    docker exec --user root ndts python3 setup.py test
    # docker exec -it ndts python3 setup.py test
fi
if [ "$?" != "0" ]; then exit 255; fi
