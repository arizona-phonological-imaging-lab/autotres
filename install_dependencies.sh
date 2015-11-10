#!/bin/sh

set -e

apt-get -y install python3-dev python3-pip g++ \
    libopenblas-dev libhdf5-dev libavbin-dev

# some day ubuntu will release a working CUDA repo
# until that day comes, we need to get it straight from nvidia
if [ ! -e 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb' ]; then
		wget 'http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
fi
    dpkg -i 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
    apt-get update
    apt-get -y install cuda
    echo 'export PATH="$PATH:/usr/local/cuda/bin"' > '/etc/profile.d/cuda.sh'
    echo '/usr/local/cuda/lib64/' > '/etc/ld.so.conf.d/cuda.conf'
    rm -f 'cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb'
    ldconfig

pip3 install -r requirements.txt

echo Autotrace dependencies successfully installed.
echo   You may have to reboot before GPU acceleration will work.
