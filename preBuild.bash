#!/bin/bash
# This file contains bash commands that will be executed at the beginning of the container build process,
# before any system packages or programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update
sudo -E apt-get install -y --no-install-recommends tzdata
sudo -E ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
sudo -E dpkg-reconfigure --frontend noninteractive tzdata
sudo -E apt-get install -y cmake zip unzip
sudo -E apt-get clean
sudo -E rm -rf /var/lib/apt/lists/*

sudo -E mkdir -p /mnt/docs
