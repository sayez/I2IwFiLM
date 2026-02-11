#!/bin/bash

years=(2011 2012 2013 2014 2015 2016 2017 2018 2019 2020)

redo_all=true

job_ids=()
for year in ${years[@]}
do
    
    echo sbatch ./scripts/aia2hmi/download_dataset.sh $year datasets/ImageTranslation_AIA_to_HMI/ $redo_all
    job_id=$(sbatch ./scripts/aia2hmi/download_dataset.sh $year datasets/ImageTranslation_AIA_to_HMI/ $redo_all | cut -d ' ' -f 4)
    job_id=$(echo $job_id | grep -o -E '[0-9]+')
    job_ids+=($job_id)

done

echo "job ids: ${job_ids[@]}"