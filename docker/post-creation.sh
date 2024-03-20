#!/bin/bash

set -e

cd /repo || exit
pip3 install -e .

# tigervnc setup
echo "123456\n" | vncpasswd
#

if nvcc --version >/dev/null 2>&1
then
    echo "nvcc is available"
else
    echo "nvcc is not available"
    echo "Manually Manually install: cuda"
    exit 0
fi
# cd /repo/repos && bash cuda_12.1.0_530.30.02_linux.run


# touch ~/.vnc/config
# mkdir -p ~/.vnc
# { echo "session=xfce4"; echo "geometry=1920x1080"; echo "localhost"; }  >> ~/.vnc/config

cd /repo || exit
source envs.sh

cd /repo/repos || exit

echo "Install three packages"

package="l4casadi"
cd $package || exit
lowercase_package=$(echo "$package" | tr '[:upper:]' '[:lower:]')
if pip3 list --format=freeze | grep -i "^$lowercase_package==" ; then
    echo "$package is already installed"
else
    echo "$package is not installed, installing now..."
    pip3 install -r requirements_build.txt
    pip3 install . --no-build-isolation
fi
cd ..


package="RLBench"
cd $package || exit
lowercase_package=$(echo "$package" | tr '[:upper:]' '[:lower:]')
if pip3 list --format=freeze | grep -i "^$lowercase_package==" ; then
    echo "$package is already installed"
else
    echo "$package is not installed, installing now..."
    pip3 install -r requirements.txt
    pip3 install . 
fi
cd ..


package="PyRep"
cd $package || exit
lowercase_package=$(echo "$package" | tr '[:upper:]' '[:lower:]')
if pip3 list --format=freeze | grep -i "^$lowercase_package==" ; then
    echo "$package is already installed"
else
    echo "$package is not installed, installing now..."
    pip3 install -r requirements.txt
    pip3 install . 
fi
cd ..
