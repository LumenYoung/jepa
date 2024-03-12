#!/bin/bash

cd ~ || exit

bash <(curl -L micro.mamba.pm/install.sh)

cd /repo || exit
pip3 install -e .
