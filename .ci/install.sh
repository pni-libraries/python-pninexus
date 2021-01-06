#!/usr/bin/env bash


docker exec  --user root ndts /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update; apt-get install -y libboost-python-dev libboost-dev'
if [ $? -ne "0" ]
then
    exit -1
fi

if [ "$1" = "2" ]; then
    echo "build python-pni"
    docker exec --user root ndts chown -R tango:tango .
    docker exec --user root ndts python setup.py build
    echo "build python3-pni docs"
    docker exec  --user root ndts python setup.py  build_sphinx
    echo "install python-pni"
    docker exec  --user root ndts python setup.py install
else
    echo "build python3-pni"
    docker exec  --user root ndts chown -R tango:tango .
    docker exec  --user root  ndts python3 setup.py build
    echo "build python3-pni docs"
    docker exec  --user root ndts python3 setup.py  build_sphinx
    echo "install python3-pni"
    docker exec  --user root ndts python3 setup.py install
fi
if [ $? -ne "0" ]
then
    exit -1
fi
