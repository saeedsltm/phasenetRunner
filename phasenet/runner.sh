#!/bin/bash

phasenet=`conda env list | grep phasenet | awk '{print $NF}'`
${phasenet}/bin/python phasenet/predict.py --model=phasenet/model/190703-214543 --data_list=tmp/mseed.csv --data_dir=tmp --result_fname=$1 --format=mseed --amplitude --response_xml=tmp/stations.xml --batch_size=1 --min_p_prob=$2 --min_s_prob=$3