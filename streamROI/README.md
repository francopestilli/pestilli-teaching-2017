## StreamROI Final Project

By [Josh Faskowitz](https://github.com/faskowit)

## Description

Final project created for P657 'The New Digital Neuroanatomy' during Spring 2017 semester at Indiana University Bloomington, Department of Psychological and Brain Sciences. This project outlines a method for using streamline tractography as a means to delimit microstructure maps to specific regions of interest. We then measure geometric similarity of these ROIs (streamROIs) by making a second-order isomorphism comparison (in similar spirit to representational similarity analysis). 

To run an example of this project, check out the `streamROI_run_through_exmple.sh` script. Make sure to alter the script to set `PYTHONBIN` to python executable (tested with python 2.7) and `PARENTWORKINGDIR` to desired output directory. This project relies on having [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), [ANTs](https://github.com/stnava/ANTs), and [Dipy](https://github.com/nipy/dipy) setup in your computing environment. 

## Abstract

Diffusion magnetic resonance imaging is a valuable method to measure the human brain’s white matter architecture in vivo and non-invasively. Using the diffusion signal, we can model complex cortical connectivity via streamline tractography methods. Additionally, we can approximate neuronal properties of white matter microstructure using advanced multi-compartment model fitting. Here, we describe a framework to jointly analyze these two diffusion modeling approaches, to measure geometric similarity of the white matter. Our approach uses the topography provided by streamline tractography to delineate areas in which to measure white matter microstructure. Using this method, we can measure microstructure overlap for streamline tracts that intersect specific areas of the cerebral cortex. Finally, we describe how derivatives of this method can be used in a multivariate analysis to compare relative geometric differences between major streamline tracts. 

