FROM debian:jessie

MAINTAINER DESY, Jan Kotanski <jankotan@gmail.com>

RUN sed -i '/jessie-updates/d' /etc/apt/sources.list
RUN apt-get -qq update && apt-get -qq install -y software-properties-common curl
RUN curl http://repos.pni-hdri.de/debian_repo.pub.gpg | apt-key add -
RUN add-apt-repository "deb http://repos.pni-hdri.de/apt/debian jessie main" -y
RUN add-apt-repository "deb  http://archive.debian.org/debian jessie-backports main non-free" -y
RUN echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf

RUN apt-get -qq update && apt-get -qq -y dist-upgrade
RUN apt-get -qq update && apt-get -qq install -y python3-sphinx apt-utils
RUN apt-get -qq install -y libpninexus2.0.0-dev libpninexus2.0.0 libh5cpp0.1.0-dev libh5cpp0.1.0 python3-numpy libhdf5-dev libboost-regex-dev libboost-program-options-dev libboost-system-dev libboost-date-time-dev libboost-filesystem-dev libblas-dev libatlas-base-dev cython3 python3-setuptools libboost-python-dev python3-numpy-abi9 g++ python3-h5py
RUN useradd -ms /bin/bash tango

ENV PKG_CONFIG_PATH=/home/tango/lib/pkgconfig
USER tango
WORKDIR /home/tango

