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

# must setup FSL up here
# FSLDIR=
# source ${FSLDIR}/etc/fslconf/fsl.sh
# IU karst: module load fsl/5.0.9

# must have ANTSPATH as well
# ANTSPATH=
# IU karst: module load ants/2.1.0

#######################################################################
# set some stuff up

PYTHONBIN=/gpfs/home/j/f/jfaskowi/Karst/anaconda2/bin/python2.7
export PARENTWORKINGDIR=${PWD}/run_through_it/
mkdir -p ${PARENTWORKINGDIR}

# will output the commands of the script into notes file
OUT=${PARENTWORKINGDIR}notes.txt
> $OUT

start=`date +%s`

#######################################################################
#######################################################################
#######################################################################
# first thing is to load the example data

mkdir -p ${PARENTWORKINGDIR}/example_data/

# based on the load_stanford_hardi script, these will be the names of
# the example data files
dwiImg=${PARENTWORKINGDIR}/example_data/stanfordHardi_dwi.nii.gz
labelsImg=${PARENTWORKINGDIR}/example_data/stanfordHardi_fsLabels.nii.gz
bvals=${PARENTWORKINGDIR}/example_data/stanfordHardi.bvals
bvecs=${PARENTWORKINGDIR}/example_data/stanfordHardi.bvecs
labelImg=${PARENTWORKINGDIR}/example_data/stanfordHardi_fsLabels.nii.gz

script='./scripts/load_stanford_hardi.py'
cmd="${PYTHONBIN} ${script} \
        -dir ${PARENTWORKINGDIR}/example_data/ \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

#######################################################################
#######################################################################
#######################################################################
# real quick make a mask with FSL

cmd="${FSLDIR}/bin/bet2 \
        ${dwiImg} \
        ${PARENTWORKINGDIR}/example_data/stanfordHardi
        -m \
        -n \
        -f 0.2 \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

maskImg=${PARENTWORKINGDIR}/example_data/stanfordHardi_mask.nii.gz

#######################################################################
#######################################################################
#######################################################################
# run a microstrucute model of your choosing
# lets do csa GFA here
# its not actually a biophysical model of WM microstructure, but we use
# it here as a proof-of-concept. One could inset a different microstructure
# map and use it in place of where 'gfaImg' variable is used. 

mkdir -p ${PARENTWORKINGDIR}/fit_microstruct/
gfaOutput=${PARENTWORKINGDIR}/fit_microstruct/stanfordHardi

script='./scripts/microstruct_mapsNmetrics_ver1.py'
cmd="${PYTHONBIN} ${script} \
        -dwi $dwiImg \
	    -mask $maskImg \
	    -bvec $bvecs \
	    -bval $bvals \
	    -output $gfaOutput \
        \
        -sh_order 6 \
        -num_peak_dirs_vol \
        -fit_csa \
        -peak_separation_angle 25 \
        -peak_relative_thresh 0.25 \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

gfaImg=${gfaOutput}_gfa_csa_sh6.nii.gz

#######################################################################
#######################################################################
#######################################################################
# lets prep for tracking a little bit

# let's extract the white matter labels from the example data
cmd="${FSLDIR}/bin/fslmaths \
        ${labelImg} \
        -thr 1 -uthr 2 \
        -bin \
        ${PARENTWORKINGDIR}/example_data/stanfordHardi_wm.nii.gz \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

wmImg=${PARENTWORKINGDIR}/example_data/stanfordHardi_wm.nii.gz 

# threshold the mask to get rid of white matter, so we can use it as the
# label mask to record connectivity from
cmd="${FSLDIR}/bin/fslmaths \
        ${labelImg} \
        -thr 3 \
        ${labelImg} \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

# get the high/low from the label file, to send to similarity mat script
cmd="${FSLDIR}/bin/fslstats \
        ${labelImg} \
        -l 0 -R \
    "
labelLow=$(eval $cmd | awk '{print $1}' )
labelHigh=$(eval $cmd | awk '{print $2}')

#######################################################################
#######################################################################
#######################################################################
# do some tractography

mkdir -p ${PARENTWORKINGDIR}/tractography/
trackOutput=${PARENTWORKINGDIR}/tractography/stanfordHardi

#tracking options
model=probabilistic
fit=csd

shOrder=6

# (-random_seed) --> yes
seedDen=4
# (-save_seed_points) --> yes

#repulsion724
sphere=repulsion724

script='./scripts/dipy_tracking_simple.py'
cmd="${PYTHONBIN} ${script} \
        -dwi $dwiImg \
	    -mask $maskImg \
	    -bvec $bvecs \
	    -bval $bvals \
	    -output ${trackOutput} \
        \
	    -tract_model ${fit} \
	    -direction_gttr ${model} \
        \
        -sh_order ${shOrder} \
	    \
	    -wm_mask ${wmImg} \
	    -rand_seed \
	    -seed_density ${seedDen} \
	    \
	    -sphere ${sphere} \
        \
        -segs ${labelImg} \
        \
        -ROIsTarg ${labelImg} \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

#######################################################################
#######################################################################
#######################################################################
# make the similarity matrices!

#inputDir=$1
#microStructMap=$2
#labelLow=$3
#labelHigh=$4
cmd="./scripts/make_similarity_mat.sh \
        ${PARENTWORKINGDIR}/tractography/ \
        ${gfaImg} \
        ${labelLow%.*} \
        ${labelHigh%.*} \
    "
echo $cmd #state the command
echo $cmd >> $OUT
eval $cmd #execute the command

#######################################################################
#######################################################################
#######################################################################

end=`date +%s`
runtime=$((end-start))
echo "runtime: $runtime"
echo "runtime: $runtime" >> $OUT
