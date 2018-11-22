#!/usr/bin/env bash


if [ $2 = "2" ]; then
    echo "install python-pni"
    # docker exec -it ndts python setup.py -q build
    # docker exec -it --user root ndts python setup.py -q build_sphinx
    docker exec -it --user root ndts /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update; apt-get install -y libboost-python-dev libboost-dev'
    docker exec -it --user root ndts python setup.py -q install
else
    echo "install python3-pni"
    if [ $1 = "debian10" ]; then
	docker exec -it --user root ndts /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update; apt-get install -y libboost-python1.62-dev libboost1.62-dev'
    else
	docker exec -it --user root ndts /bin/sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update; apt-get install -y libboost-python-dev libboost-dev'
    fi
    # docker exec -it ndts python3 setup.py -q build
    # docker exec -it --user root ndts python3 setup.py -q build_sphinx
    docker exec -it --user root ndts python3 setup.py -q install
fi
if [ $? -ne "0" ]
then
    exit -1
fi
