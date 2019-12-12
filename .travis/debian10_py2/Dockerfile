FROM debian:buster

MAINTAINER DESY, Jan Kotanski <jankotan@gmail.com>
RUN /bin/bash -c 'echo "deb http://ftp.de.debian.org/debian buster main" > /etc/apt/sources.list.d/debian10de.list'
RUN apt-get -qq update && apt-get -qq install -y libterm-readline-gnu-perl software-properties-common coreutils gnupg2 procps apt-utils curl apt-transport-https gnupg2 ca-certificates
RUN curl http://repos.pni-hdri.de/debian_repo.pub.gpg | apt-key add -
RUN add-apt-repository "deb http://repos.pni-hdri.de/apt/debian buster main" -y
RUN apt-get -qq update && apt-get -qq -y dist-upgrade
RUN apt-get -qq update && apt-get -qq install -y python-sphinx  apt-utils net-tools
RUN apt-get -qq install -y libpnicore1.1.0-dev libpnicore1.1.0 libpniio1.2.7-dev libpniio1.2.7 libh5cpp0.3.2-dev libh5cpp0.3.2 python-numpy libhdf5-dev libboost-regex-dev libboost-program-options-dev libboost-system-dev libboost-date-time-dev libboost-filesystem-dev libblas-dev libatlas-base-dev cython python-setuptools libboost-python-dev python-numpy-abi9 g++ python-h5py hdf5-plugin-bshuf hdf5-plugin-bz2 hdf5-plugin-lz4 hdf5-plugins-common
RUN useradd -ms /bin/bash tango
RUN apt-get -qq install -y adduser

ENV PKG_CONFIG_PATH=/home/tango/lib/pkgconfig
USER tango
WORKDIR /home/tango
