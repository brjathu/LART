#!/bin/bash

# Create the data directory
mkdir data
mkdir data/_TMP
mkdir data/bad_files


# Download the data
wget https://www.dropbox.com/s/gqjfd6f4e9lvl76/ava_val.tar.gz -P data/
wget https://www.dropbox.com/s/bib5lzrsv7res1b/ava_train-aa.tar.gz -P data/
wget https://www.dropbox.com/s/79yimgh6pv06krs/ava_train-ab.tar.gz -P data/
wget https://www.dropbox.com/s/rpvh6h4vzlmf5y4/ava_train-ac.tar.gz -P data/
wget https://www.dropbox.com/s/11hce70tn6twow7/ava_train-ad.tar.gz -P data/
wget https://www.dropbox.com/s/5khqhta387xmnqu/kinetics_train-aa.tar.gz -P data/
wget https://www.dropbox.com/s/mmzspxq3fzbhleg/kinetics_train-ab.tar.gz -P data/
wget https://www.dropbox.com/s/6j5eqw9l26iylek/kinetics_train-ac.tar.gz -P data/
wget https://www.dropbox.com/s/mo59q8rj72n1o5x/kinetics_train-ad.tar.gz -P data/
wget https://www.dropbox.com/s/qxvp4x4lfdxdscv/kinetics_train-ae.tar.gz -P data/
wget https://www.dropbox.com/s/5c8uv3yyi51w5tq/kinetics_train-af.tar.gz -P data/
wget https://www.dropbox.com/s/xsufg8fvsb631af/kinetics_train-ag.tar.gz -P data/
wget https://www.dropbox.com/s/1kwc78seab180qc/kinetics_train-ah.tar.gz -P data/
wget https://www.dropbox.com/s/suaic1vbsgwkab5/kinetics_train-ai.tar.gz -P data/
wget https://www.dropbox.com/s/c6ji8uxo333yus2/kinetics_train-aj.tar.gz -P data/
wget https://www.dropbox.com/s/c4rm4v7mcc7har3/kinetics_train-ak.tar.gz -P data/
wget https://www.dropbox.com/s/p4n8bszdojmxj1q/kinetics_train-al.tar.gz -P data/
wget https://www.dropbox.com/s/60mud6lizsyj0n4/kinetics_train-am.tar.gz -P data/

# Extract the data
tar -xvzf data/ava_val.tar.gz -C data/
tar -xvzf data/ava_train-aa.tar.gz -C data/
tar -xvzf data/ava_train-ab.tar.gz -C data/
tar -xvzf data/ava_train-ac.tar.gz -C data/
tar -xvzf data/ava_train-ad.tar.gz -C data/
tar -xvzf data/kinetics_train-aa.tar.gz -C data/
tar -xvzf data/kinetics_train-ab.tar.gz -C data/
tar -xvzf data/kinetics_train-ac.tar.gz -C data/
tar -xvzf data/kinetics_train-ad.tar.gz -C data/
tar -xvzf data/kinetics_train-ae.tar.gz -C data/
tar -xvzf data/kinetics_train-af.tar.gz -C data/
tar -xvzf data/kinetics_train-ag.tar.gz -C data/
tar -xvzf data/kinetics_train-ah.tar.gz -C data/
tar -xvzf data/kinetics_train-ai.tar.gz -C data/
tar -xvzf data/kinetics_train-aj.tar.gz -C data/
tar -xvzf data/kinetics_train-ak.tar.gz -C data/
tar -xvzf data/kinetics_train-al.tar.gz -C data/
tar -xvzf data/kinetics_train-am.tar.gz -C data/
