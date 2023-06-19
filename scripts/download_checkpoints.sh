#!/bin/bash

# Create the directory
mkdir logs

# Download the checkpoints
wget https://www.dropbox.com/s/aja2sfo9ge8hqxf/lart_hiera_001.tar.gz

# Extract the checkpoints
tar -xzf lart_hiera_001.tar.gz

# Remove the tar file
rm lart_hiera_001.tar.gz