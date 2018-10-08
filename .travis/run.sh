#!/usr/bin/env bash

if [ $1 = "2" ]; then
    echo "run python-pni"
    # docker exec -it --user root ndts python setup.py test
    docker exec -it ndts python setup.py test
else
    echo "run python3-pni"
    # docker exec -it --user root ndts python3 setup.py test
    docker exec -it ndts python3 setup.py test
fi
if [ $? -ne "0" ]
then
    exit -1
fi
