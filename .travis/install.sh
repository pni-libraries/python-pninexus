#!/usr/bin/env bash


if [ $2 = "2" ]; then
    echo "install python-pni"
    # docker exec -it --user root ndts python setup.py -q build
    # docker exec -it --user root ndts python setup.py -q build_sphinx
    docker exec -it --user root ndts python setup.py -q install
else
    echo "install python3-pni"
    # docker exec -it --user root ndts python3 setup.py -q build
    # docker exec -it --user root ndts python3 setup.py -q build_sphinx
    docker exec -it --user root ndts python3 setup.py -q install
fi
if [ $? -ne "0" ]
then
    exit -1
fi