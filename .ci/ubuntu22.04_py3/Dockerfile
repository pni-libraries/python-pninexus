FROM ubuntu:22.04

MAINTAINER DESY, Jan Kotanski <jankotan@gmail.com>

RUN apt-get update
RUN apt-get install -y base-passwd apt-utils dialog -y
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get -qq update && apt-get -qq install -y libterm-readline-gnu-perl software-properties-common coreutils gnupg2 procps curl apt-transport-https gnupg2 ca-certificates wget
RUN curl -s http://repos.pni-hdri.de/debian_repo.pub.gpg  | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/debian-hdri-repo.gpg --import
RUN chmod 644 /etc/apt/trusted.gpg.d/debian-hdri-repo.gpg
RUN add-apt-repository "deb http://repos.pni-hdri.de/apt/debian jammy main" -y
RUN apt-get -qq update && export DEBIAN_FRONTEND=noninteractive && apt-get -qq dist-upgrade
RUN apt-get -qq update && export DEBIAN_FRONTEND=noninteractive && apt-get -qq install -y python3-sphinx apt-utils net-tools 
RUN apt-get -qq install -y libpninexus3.0.2-dev libpninexus3.0.2 libh5cpp0.5.2-dev libh5cpp0.5.2 python3-numpy libhdf5-dev libboost-regex-dev libboost-program-options-dev libboost-system-dev libboost-date-time-dev libboost-filesystem-dev libblas-dev libatlas-base-dev cython3 python3-setuptools libboost-python-dev python3-numpy-abi9 g++ python3-h5py
RUN apt-get -qq install -y hdf5-plugin-bshuf hdf5-plugin-bz2 hdf5-plugin-lz4 python3-pytest git cmake doxygen python3-sphinx libboost-all-dev
RUN useradd -ms /bin/bash tango

# # install latest h5cpp
# RUN mkdir -p /opt/h5cpp
# RUN mkdir -p /h5cpp-src
# RUN cd /h5cpp-src && git clone https://github.com/ess-dmsc/h5cpp && mkdir build
# RUN cd /h5cpp-src/build && cmake -DCMAKE_INSTALL_PREFIX=/opt/h5cpp -DCMAKE_BUILD_TYPE=Release -DH5CPP_CONAN=DISABLE -DH5CPP_DISABLE_TESTS=True  ../h5cpp && make install -j4
# ENV CPLUS_INCLUDE_PATH=/opt/h5cpp/include
# ENV LD_LIBRARY_PATH=/opt/h5cpp/lib
# ENV H5CPP=/opt/h5cpp/lib/cmake/h5cpp-0.5
# ENV H5CPP_LOCAL_PATH=/opt/h5cpp
# ENV HDF5_HL_LOCAL_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial/

# # install latest linpninexus
# RUN mkdir -p /opt/libpninexus
# RUN mkdir -p /libpninexus-src
# RUN cd /libpninexus-src && git clone https://github.com/pni-libraries/libpninexus && mkdir build
# RUN cd /libpninexus-src/libpninexus && git submodule init && git pull --recurse-submodules
# RUN cd /libpninexus-src/build && cmake -Dh5cpp_DIR=$H5CPP  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/h5cpp/   -DPNINEXUS_CONAN=DISABLE ../libpninexus  && make -j4 && make install -j4
# ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/opt/libpninexus/include
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libpninexus/lib
# ENV PNINEXUS_LOCAL_PATH=/opt/libpninexus

ENV PKG_CONFIG_PATH=/home/tango/lib/pkgconfig
USER tango
WORKDIR /home/tango

