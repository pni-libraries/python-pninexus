#!/usr/bin/env bash


if [ "$1" = "2" ]; then
    echo "install python-pninexus"
    docker exec --user root ndts chown -R tango:tango .
    docker exec  --user root ndts python setup.py install
    echo "build python3-pninexus docs"
    docker exec  --user root ndts python setup.py  build_sphinx
else
    echo "install python3-pninexus"
    docker exec  --user root ndts chown -R tango:tango .
    docker exec  --user root ndts python3 setup.py install
    echo "build python3-pninexus docs"
    docker exec  --user root ndts python3 setup.py  build_sphinx
fi
if [ $? -ne "0" ]
then
    exit -1
fi
