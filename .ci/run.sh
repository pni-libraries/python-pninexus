#!/usr/bin/env bash

echo "run python3-pninexus tests"
docker exec ndts python3 -m pytest test
if [ "$?" != "0" ]; then exit 255; fi
