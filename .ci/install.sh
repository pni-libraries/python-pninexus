#!/usr/bin/env bash


echo "install python3-pninexus"
docker exec  --user root ndts chown -R tango:tango .
docker exec ndts mkdir -p build
if [ "$?" != "0" ]; then exit 255; fi
if [ "$1" = "ubuntu22.04" ] || [ "$1" = "debian11"] ; then
    docker exec --user root ndts python3 -m pip install . -vv  --no-deps --no-build-isolation
else
    docker exec --user root ndts python3 -m pip install . -vv  --no-deps --no-build-isolation  --break-system-packages --root-user-action=ignore
fi
if [ "$?" != "0" ]; then exit 255; fi
echo "build python3-pninexus docs"
docker exec ndts mkdir -p doc/sphinx/_static
docker exec ndts sphinx-build doc/sphinx/ build/sphinx/html

if [ "$1" != "ubuntu26.04" ] && [ "$?" != "0" ]; then exit 255; fi

docker exec  --user root ndts rm -rf src/pninexus.egg-info
if [ "$?" != "0" ]; then exit 255; fi
