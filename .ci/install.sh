#!/usr/bin/env bash


if [ "$1" = "2" ]; then
    echo "install python-pninexus"
    docker exec --user root ndts chown -R tango:tango .
    docker exec ndts python setup.py build
    if [ "$?" != "0" ]; then exit 255; fi
    docker exec  --user root ndts python setup.py install
    if [ "$?" != "0" ]; then exit 255; fi
    echo "build python3-pninexus docs"
    docker exec ndts python setup.py  build_sphinx
else
    echo "install python3-pninexus"
    docker exec  --user root ndts chown -R tango:tango .
    docker exec ndts python3 setup.py build
    if [ "$?" != "0" ]; then exit 255; fi
    docker exec  --user root ndts python3 setup.py install
    if [ "$?" != "0" ]; then exit 255; fi
    echo "build python3-pninexus docs"
    docker exec ndts python3 setup.py  build_sphinx
fi
if [ "$?" != "0" ]; then exit 255; fi

docker exec  --user root ndts rm -rf src/pninexus.egg-info
if [ "$?" != "0" ]; then exit 255; fi
