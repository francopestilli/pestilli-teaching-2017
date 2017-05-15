#!/bin/bash

<<'COMMENT'
josh faskowitz
Indiana University
Computational Cognitive Neurosciene Lab

University of Southern California
Imaging Genetics Center

Copyright (c) 2017 Josh Faskowitz
See LICENSE file for license
COMMENT

# the PARENTWORKINGDIR variabe is exported from script calling
# this script
workingDir=${PARENTWORKINGDIR}/similarity_mats/

# read in data here
inputDir=$1
microStructMap=$2
labelLow=$3
labelHigh=$4

###########################################################

#If not created yet, let's create a new output folder
mkdir -p $workingDir

#go into the folder where the script should be run
cd $workingDir
echo "CHANGING DIRECTORY into $workingDir"

# let's write out what script does into notes file
OUT=notes.txt
> $OUT

###########################################################

image_labels=($(seq ${labelLow} ${labelHigh}))

###########################################################

# loop it to make the normDens
for (( idx=1 ; idx<=${#image_labels[@]} ; idx++ ))
do

    tmp_dens=$(ls ${inputDir}/*lab${image_labels[${idx}-1]}_dens.nii.gz 2>/dev/null)

    if [[ ! -e ${tmp_dens} ]]
    then
        continue
    fi

    # lets normalize better
    # read the 95% and 5% percentile values
    robMax=$(${FSLDIR}/bin/fslstats ${tmp_dens} -P 95)
    robMin=$(${FSLDIR}/bin/fslstats ${tmp_dens} -P 5)

    # get difference between these high + low vals
    divBy=$(echo "${robMax} - ${robMin}" | bc)

    # bring down values higher than 95% percentile value 
    # to the 95% value. subtract min value and normalize 
    # from 0-1
    cmd="${FSLDIR}/bin/fslmaths \
            ${tmp_dens} -min ${robMax} \
            -sub ${robMin} -thr 0 \
            -div ${divBy} \
            ${workingDir}/tmpNorm${idx}.nii.gz \
        "
    echo $cmd #state the command
    log $cmd >> $OUT
    eval ${cmd}

    # multiply microStructMap by the stream density %
    cmd="${FSLDIR}/bin/fslmaths \
            ${microStructMap} \
            -mul ${workingDir}/tmpNorm${idx}.nii.gz\
            ${workingDir}/tmpNormDens${idx}.nii.gz \
        "
    echo $cmd #state the command
    log $cmd >> $OUT
    eval ${cmd}

done

# the ANTs command
# MeasureImageSimilarity ImageDimension whichmetric image1.ext image2.ext {logfile} {outimage.ext}  {target-value}   {epsilon-tolerance}
# outimage (Not Implemented for MI yet)  and logfile are optional  
# target-value and epsilon-tolerance set goals for the metric value -- if the metric value is within epsilon-tolerance of the target-value, then the test succeeds  Metric 0 - MeanSquareDifference, 1 - Cross-Correlation, 2-Mutual Information , 3-SMI 

# initialize rdm 
outFile_1=${workingDir}/mi_steamROI_aparc+aseg_rdm.csv
> ${outFile_1}

# TOO SLOW FOR NOW
#outFile_2=${workingDir}/cc_aparc+aseg_rdm.csv
#> ${outFile_2}

outFile_3=${workingDir}/msq_densitiy_aparc+aseg_rdm.csv
> ${outFile_3}

outFile_4=${workingDir}/mi_densitiy_aparc+aseg_rdm.csv
> ${outFile_4}

# row 
for (( idx=1 ; idx<=${#image_labels[@]} ; idx++))
do
    simRow_mi=''
    #simRow_cc=''
    simRow_msq_d=''
    simRow_mi_d=''    

    # colum
    for (( jdx=1 ; jdx<=${#image_labels[@]} ; jdx++))
    do

        tmp_dens1=$(ls ${inputDir}/*lab${image_labels[${idx}-1]}_dens.nii.gz 2>/dev/null)
        tmp_dens2=$(ls ${inputDir}/*lab${image_labels[${jdx}-1]}_dens.nii.gz 2>/dev/null)

        # if row number greater than colum, we want 0
        # because we want upper triangle
        if [[ ${jdx} -lt ${idx} ]]
        then
            sim_mi=0
            #sim_cc=0
            sim_msq_d=0
            sim_mi_d=0
    
        elif [[ ${jdx} -eq ${idx} ]]
        then
            sim_mi=0
            #sim_cc=0
            sim_msq_d=0
            sim_mi_d=0

        elif [[ ! -e ${tmp_dens1} ]] || [[ ! -e ${tmp_dens2} ]]
        then
            sim_mi=0
            #sim_cc=0
            sim_msq_d=0
            sim_mi_d=0

        else

            # similarity mi
            # (returns the negative mutual information)
            cmd="${ANTSPATH}/MeasureImageSimilarity 3 2 \
                    ${workingDir}/tmpNormDens${idx}.nii.gz ${workingDir}/tmpNormDens${jdx}.nii.gz \
                    ${workingDir}/tmpLog.txt \
                "
            echo $cmd #state the command
            log $cmd >> $OUT
            eval ${cmd}

            # multiply by -1 to get positive mutual information
            sim_mi=$(cat ${workingDir}/tmpLog.txt | awk '{print (-1 * $NF)}')
            rm ${workingDir}/tmpLog.txt

            #########################
            #########################

            # similarity cc
            #cmd="${ANTSPATH}/MeasureImageSimilarity 3 1 \
            #        ${workingDir}/tmpNormDens${idx}.nii.gz ${workingDir}/tmpNormDens${jdx}.nii.gz \
            #        ${workingDir}/tmpLog.txt \
            #    "
            #echo $cmd #state the command
            #log $cmd >> $OUT
            #eval ${cmd}

            #sim_cc=$(cat ${workingDir}/tmpLog.txt | awk '{print (1 * $NF)}')
            #rm ${workingDir}/tmpLog.txt

            #########################
            #########################

            # note: here we measure against tmpNorm 

            # similarity msq
            cmd="${ANTSPATH}/MeasureImageSimilarity 3 0 \
                    ${workingDir}/tmpNorm${idx}.nii.gz ${workingDir}/tmpNorm${jdx}.nii.gz \
                    ${workingDir}/tmpLog.txt \
                "
            echo $cmd #state the command
            log $cmd >> $OUT
            eval ${cmd}

            sim_msq_d=$(cat ${workingDir}/tmpLog.txt | awk '{print (1 * $NF)}')
            rm ${workingDir}/tmpLog.txt

            #########################
            #########################

            # note: here we measure against tmpNorm 

            # (returns the negative mutual information)
            cmd="${ANTSPATH}/MeasureImageSimilarity 3 2 \
                    ${workingDir}/tmpNorm${idx}.nii.gz ${workingDir}/tmpNorm${jdx}.nii.gz \
                    ${workingDir}/tmpLog.txt \
                "
            echo $cmd #state the command
            log $cmd >> $OUT
            eval ${cmd}

            sim_mi_d=$(cat ${workingDir}/tmpLog.txt | awk '{print (1 * $NF)}')
            rm ${workingDir}/tmpLog.txt

        fi

        simRow_mi="${simRow_mi}${sim_mi},"
        #simRow_cc="${simRow_cc}${sim_cc},"
        simRow_msq_d="${simRow_msq_d}${sim_msq_d},"
        simRow_mi_d="${simRow_mi_d}${sim_mi_d},"

    done

    echo ${simRow_mi} | sed 's,.$,,' >> ${outFile_1}
    #echo ${simRow_cc} | sed 's,.$,,' >> ${outFile_2}
    echo ${simRow_msq_d} | sed 's,.$,,' >> ${outFile_3}
    echo ${simRow_mi_d} | sed 's,.$,,' >> ${outFile_4}

done

# optional cleanup
ls ${workingDir}/tmpNorm*.nii.gz && rm ${workingDir}/tmpNorm*.nii.gz

###########################################################
###########################################################
###########################################################

