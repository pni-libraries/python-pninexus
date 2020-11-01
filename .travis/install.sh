#!/usr/bin/env bash


docker exec -it --user root ndts /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update; apt-get install -y libboost-python-dev libboost-dev'
if [ $? -ne "0" ]
then
    exit -1
fi

if [ $2 = "2" ]; then
    echo "install python-pni"
    # docker exec -it ndts python setup.py -q build
    # docker exec -it --user root ndts python setup.py -q build_sphinx
    docker exec -it --user root ndts chown -R tango:tango .
    docker exec -it --user root ndts python setup.py build
    docker exec -it --user root ndts python setup.py build_sphinx
    docker exec -it --user root ndts python setup.py install
else
    echo "install python3-pni"
    # docker exec -it ndts python3 setup.py -q build
    # docker exec -it --user root ndts python3 setup.py -q build_sphinx
    docker exec -it --user root ndts chown -R tango:tango .
    docker exec -it --user root ndts python3 setup.py build
    docker exec -it --user root ndts python3 setup.py build_sphinx
    docker exec -it --user root ndts python3 setup.py install
fi
if [ $? -ne "0" ]
then
    exit -1
fi
