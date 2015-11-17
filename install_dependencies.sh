#!/bin/sh

set -e

apt-get -y install python3-dev python3-pip \
    g++ gfortran libopenblas-dev liblapack-dev \
    libhdf5-dev libavbin0

# some day ubuntu will release a working CUDA repo
# until that day comes, we need to get it straight from nvidia
if [ ! -e 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb' ]; then
		wget 'http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
fi
    dpkg -i 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
    apt-get update
    apt-get -y install cuda-driver-dev-7-5
    apt-get -y install nvidia-opencl-icd-352
    apt-get -y install cuda-drivers
    apt-get -y install cuda-runtime-7.5
    apt-get -y install cuda-7.5
    echo 'export PATH="$PATH:/usr/local/cuda/bin"' > '/etc/profile.d/cuda.sh'
    echo '/usr/local/cuda/lib64/' > '/etc/ld.so.conf.d/cuda.conf'
    rm -f 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
    ldconfig

pip3 install -e . #r requirements.txt

echo Autotrace dependencies successfully installed.
echo   You may have to reboot before GPU acceleration will work.
