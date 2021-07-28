FROM ubuntu:21.04

MAINTAINER DESY, Jan Kotanski <jankotan@gmail.com>

RUN apt-get -qq update && export DEBIAN_FRONTEND=noninteractive && apt-get -qq install -y software-properties-common curl
RUN curl http://repos.pni-hdri.de/debian_repo.pub.gpg | apt-key add -
RUN add-apt-repository "deb http://repos.pni-hdri.de/apt/debian hirsute main" -y
RUN apt-get -qq update && export DEBIAN_FRONTEND=noninteractive && apt-get -qq dist-upgrade
RUN apt-get -qq update && export DEBIAN_FRONTEND=noninteractive && apt-get -qq install -y python3-sphinx apt-utils net-tools 
RUN apt-get -qq install -y libpninexus2.0.0-dev libpninexus2.0.0 libh5cpp0.4.1-dev libh5cpp0.4.1 python3-numpy libhdf5-dev libboost-regex-dev libboost-program-options-dev libboost-system-dev libboost-date-time-dev libboost-filesystem-dev libblas-dev libatlas-base-dev cython3 python3-setuptools libboost-python-dev python3-numpy-abi9 g++ python3-h5py
RUN apt-get -qq install -y hdf5-plugin-bshuf hdf5-plugin-bz2 hdf5-plugin-lz4
RUN useradd -ms /bin/bash tango

ENV PKG_CONFIG_PATH=/home/tango/lib/pkgconfig
ENV HDF5_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/hdf5/plugins
USER tango
WORKDIR /home/tango

