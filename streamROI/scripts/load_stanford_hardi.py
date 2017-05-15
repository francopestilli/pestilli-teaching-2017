import os
import sys
import argparse
import nibabel as nib

__author__ = 'jfaskowitz'

"""
josh faskowitz
Indiana University
Computational Cognitive Neurosciene Lab

University of Southern California
Imaging Genetics Center

Copyright (c) 2017 Josh Faskowitz
See LICENSE file for license
"""

def main():

    argParseObj = argparse.ArgumentParser(description="Load stanford hardi dataset")

    # this is the whole dwi with all the volumes yo
    argParseObj.add_argument('-dir', help="the path where you want"
                                          "this stuff to be dumped",
                             type=str, nargs='?', required=True)
    args = argParseObj.parse_args()

    outputDir = args.dir

    from dipy.data import read_stanford_labels
    hardi_img, gtab, label_img = read_stanford_labels()

    hardi_data = hardi_img.get_data()
    hardi_write = nib.Nifti1Image(hardi_data, hardi_img.get_affine())
    outName = ''.join([outputDir, 'stanfordHardi_dwi.nii.gz'])
    nib.save(hardi_write, outName)

    label_data = label_img.get_data()
    label_write = nib.Nifti1Image(label_data, label_img.get_affine())
    outName = ''.join([outputDir, 'stanfordHardi_fsLabels.nii.gz'])
    nib.save(label_write, outName)

    # from dipy.data import read_stanford_pve_maps
    # csf_img , gm_img , wm_img = read_stanford_pve_maps()

    from dipy.external import fsl
    outName = ''.join([outputDir, 'stanfordHardi.'])
    fsl.write_bvals_bvecs(gtab.bvals, gtab.bvecs, prefix=outName)

if __name__ == '__main__':
    main()
