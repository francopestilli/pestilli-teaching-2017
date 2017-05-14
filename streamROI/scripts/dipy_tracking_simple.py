import sys
import ntpath
import numpy as np
import nibabel as nib
import h5py
# import random
from dipy_tracking_args import *

__author__ = 'jfaskowitz'

'''
josh faskowitz
Indiana University
Computational Cognitive Neurosciene Lab

University of Southern California
Imaging Genetics Center

Copyright (c) 2017 Josh Faskowitz
See LICENSE file for license
'''

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def main():
    cmdLineArgs = GetParams()
    cmdLineArgs.do_work()

    for attr, value in GetParams.__dict__.iteritems():
        print attr, value

    # flush python stuff to the screen
    sys.stdout.flush()

    # if the output string does not exist, just name it 'output'
    if not cmdLineArgs.output_:
        cmdLineArgs.output_ = 'output_'

    # convert the strings into images we can do stuff with in dipy yo!
    dwiImage = nib.load(cmdLineArgs.dwi_)
    maskImage = nib.load(cmdLineArgs.mask_)

    wmMaskImage = ''
    if cmdLineArgs.wmMask_:
        wmMaskImage = nib.load(cmdLineArgs.wmMask_)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ BVALS AND BVECS ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # read the bvals and bvecs into data artictecture that dipy understands
    from dipy.io import read_bvals_bvecs
    bvalText, bvecText = read_bvals_bvecs(cmdLineArgs.bval_, cmdLineArgs.bvec_)

    # need to create the gradient table yo
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvalText, bvecText, b0_threshold=5)

    # get the data from all the images yo
    dwiData = dwiImage.get_data()
    maskData = maskImage.get_data()

    mask_affine = maskImage.get_affine()
    dwi_affine = dwiImage.get_affine()

    # and then mask the dwiData
    from dipy.segment.mask import applymask
    # apply the provided mask to the dwi data
    dwiData = applymask(dwiData, maskData)

    wmMaskData = None
    # check to see if wmMask provided
    if cmdLineArgs.wmMask_:
        wmMaskData = wmMaskImage.get_data()

    print('dwiData shape: (%d, %d, %d, %d)' % dwiData.shape)
    print('\nbvecs:{0}'.format(bvecText))
    print('\nbvals:{0}\n'.format(bvalText))

    # TODO might change the sphere based on user input...
    print('getting sphere')
    from dipy.data import get_sphere
    sphere = get_sphere(cmdLineArgs.sphere_)

    sys.stdout.flush()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RANDOM SEED and DENSITY ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO give user option to save seed points as npz for future usage

    seed_points = None

    if not cmdLineArgs.seedPointsFile_:

        from dipy.tracking import utils
        if cmdLineArgs.wmMask_:

            if cmdLineArgs.dirGttr_ == 'eudx':
                print('eudx hack')

                wmMaskData = applymask(wmMaskData, maskData)

            # this is the common usage
            if cmdLineArgs.randSeed_:
                # make the seeds randomly yo
                print("using a wm mask for random seeds with density of {}\n".format(str(cmdLineArgs.seedDensity_)))

                from dipy.tracking.utils import random_seeds_from_mask
                seed_points = random_seeds_from_mask(wmMaskData,
                                                     seeds_count=cmdLineArgs.seedDensity_,
                                                     affine=mask_affine)
            else:
                # make the seeds yo
                print(
                    "using a wm mask for NON-random seeds with a density of {}\n".format(str(cmdLineArgs.seedDensity_)))

                seed_points = utils.seeds_from_mask(wmMaskData,
                                                    density=int(cmdLineArgs.seedDensity_),
                                                    affine=mask_affine)
        else:
            # seed from the binary mask yo
            seed_points = utils.seeds_from_mask(maskData,
                                                density=1,
                                                affine=mask_affine)

    else:
        # load the seed points, and lets hope the user knows what they are doing!
        # seed points must be in same affine space as what woulda been generated

        print("loading seed points from file: {}".format(str(cmdLineArgs.seedPointsFile_)))
        sys.stdout.flush()

        seedFile = np.load(cmdLineArgs.seedPointsFile_[0])
        seed_points = seedFile['arr_0']

    sys.stdout.flush()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CLASSIFIER YO ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    classifier = None

    # ###### NEW TISSUES CLASSIFIER STUFF ########
    if cmdLineArgs.actClasses_:
        # do the ACT Tissue Classifier
        from dipy.tracking.local import ActTissueClassifier

        print("making the classifier from your segs yo\n")

        # csf, gm, wm
        csfImage = nib.load(cmdLineArgs.actClasses_[0])
        csfData = csfImage.get_data()

        gmImage = nib.load(cmdLineArgs.actClasses_[1])
        gmData = gmImage.get_data()

        wmImage = nib.load(cmdLineArgs.actClasses_[2])
        wmData = wmImage.get_data()

        # make a background
        background = np.ones(maskData.shape)
        background[(gmData + wmData + csfData) > 0] = 0

        include_map = gmData
        include_map[background > 0] = 1

        exclude_map = csfData

        classifier = ActTissueClassifier(include_map, exclude_map)

    else:

        print("making some FA yo for the threshold classifier\n")

        import dipy.reconst.dti as dti
        from dipy.reconst.dti import fractional_anisotropy

        tensor_model = dti.TensorModel(gtab, fit_method="WLS")
        tenfit = tensor_model.fit(dwiData, mask=maskData)

        from dipy.tracking.local import ThresholdTissueClassifier
        FA = fractional_anisotropy(tenfit.evals)

        # just saving the FA image yo
        FA[np.isnan(FA)] = 0

        # also we can clip values outside of 0 and 1
        FA = np.clip(FA, 0, 1)

        '''
        fa_img = nib.Nifti1Image(FA.astype(np.float32), dwiImage.get_affine())

        #make the output name yo
        fa_outputname = ''.join([outputname,'_fa.nii.gz'])
        print('The output FA name is: {}'.format(fa_outputname))

        #same this FA
        nib.save(fa_img, fa_outputname)

        print("making the FA classifier yo with a FA thresh of {}\n".format(cmdLineArgs.faClassify_))
        '''

        classifier = ThresholdTissueClassifier(FA, float(cmdLineArgs.faClassify_))

    sys.stdout.flush()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIFFUSION / FIBER MODELS ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # do this so my linter doesnt complain about acessing variables
    # before they are assigned. i know, silly.
    shcoeffs = None
    peakDirections = None

    # must make this a bit if elif else statement
    if cmdLineArgs.tractModel_ == 'csd':

        print('using the csd model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if not cmdLineArgs.coeffFile_:

            print("making the CSD ODF model with sh of {}\n".format(cmdLineArgs.shOrder_))

            from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                               auto_response)

            # get the response yo.
            response, ratio = auto_response(gtab, dwiData, roi_radius=10, fa_thr=0.7)

            print("the response for the csd is:\n{0}\nwith a ratio of:\n{1}\n".format(response, ratio))

            # chaning the sh order to 6 yo
            csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=int(cmdLineArgs.shOrder_))

            print("making the CSD fit yo\n")
            sys.stdout.flush()

            csd_fit = csd_model.fit(dwiData, mask=maskData)
            shcoeffs = csd_fit.shm_coeff

        else:  # i.e. we already have generated the coeffs

            print("loading coeffs from file: {}".format(str(cmdLineArgs.coeffFile_)))
            sys.stdout.flush()

            # TODO new load format
            # coeffsFile = np.load(cmdLineArgs.coeffFile_)
            # shcoeffs = coeffsFile['arr_0']

            # the new format yo
            coeffsFile = h5py.File(cmdLineArgs.coeffFile_, 'r')
            shcoeffs = np.array(coeffsFile['PAM/coeff'])

    elif cmdLineArgs.tractModel_ == 'csa':

        print('using the csa model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSA FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model

        print("generating csa model. the sh order is {}".format(cmdLineArgs.shOrder_))

        csa_model = CsaOdfModel(gtab, sh_order=cmdLineArgs.shOrder_)
        csa_peaks = peaks_from_model(model=csa_model,
                                     data=dwiData,
                                     sphere=sphere,
                                     relative_peak_threshold=0.5,
                                     min_separation_angle=25,
                                     mask=maskData,
                                     return_sh=True,
                                     parallel=False)

        # TODO fix this
        shcoeffs = csa_peaks.shm_coeff

    elif cmdLineArgs.tractModel_ == 'sparse':

        # TODO gotta perhaps implement this one
        pass

    else:  # this is for DTI model yo.

        print('using the dti model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ TENSOR FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print("making some FA yo")
        sys.stdout.flush()

        import dipy.reconst.dti as dti
        from dipy.reconst.dti import fractional_anisotropy

        tensor_model = dti.TensorModel(gtab, fit_method="WLS")

        tenfit = tensor_model.fit(dwiData, mask=maskData)
        FA = fractional_anisotropy(tenfit.evals)

        # just saving the FA image yo
        FA[np.isnan(FA)] = 0

        # also we can clip values outside of 0 and 1
        FA = np.clip(FA, 0, 1)
        fa_img = nib.Nifti1Image(FA.astype(np.float32), dwiImage.get_affine())

        # make the output name yo
        fa_outputname = ''.join([cmdLineArgs.output_, '_fa.nii.gz'])
        print('The output FA name is: {}'.format(fa_outputname))
        sys.stdout.flush()

        # same this FA
        nib.save(fa_img, fa_outputname)

        if cmdLineArgs.dirGttr_ != 'eudx':

            print('making dti tractography with localTracking')
            sys.stdout.flush()

            from dipy.reconst.peaks import peaks_from_model
            peakDirections = peaks_from_model(tensor_model,
                                              data=dwiData.astype(np.float_),
                                              sphere=sphere,
                                              relative_peak_threshold=0.5,
                                              min_separation_angle=25,
                                              mask=maskData,
                                              parallel=False)

        elif cmdLineArgs.dirGttr_ == 'eudx':

            print('making dti tractography with eudx')

            # make some evecs yo...
            evecs = tenfit.evecs.astype(np.float32)
            evecs = applymask(evecs, maskData)

            from dipy.reconst.dti import quantize_evecs

            peakDirections = quantize_evecs(evecs=evecs, odf_vertices=sphere.vertices)

    sys.stdout.flush()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIRECTION GETTR ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    directionGetter = None

    print("generating the direction gettr\n")
    sys.stdout.flush()

    if (cmdLineArgs.tractModel_ == 'csa') or (cmdLineArgs.tractModel_ == 'csd'):

        if cmdLineArgs.dirGttr_ == 'deterministic':

            from dipy.direction import DeterministicMaximumDirectionGetter

            print("determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=30.0,
                                                                               sphere=sphere)
        elif cmdLineArgs.dirGttr_ == 'probabilistic':

            from dipy.direction import ProbabilisticDirectionGetter

            print("prob direction gettr\n")
            directionGetter = ProbabilisticDirectionGetter.from_shcoeff(shcoeffs,
                                                                        max_angle=30.0,
                                                                        sphere=sphere)

        else:

            from dipy.direction import DeterministicMaximumDirectionGetter

            print("you forog to to specify a dir gtter, so we will use determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=30.0,
                                                                               sphere=sphere)

    else:
        # dont need a deter or prob dir getter

        directionGetter = peakDirections

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ STREAMLINES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    streamlines = None

    print("making streamlines yo\n")
    sys.stdout.flush()

    if cmdLineArgs.dirGttr_ != 'eudx':

        from dipy.tracking.local import LocalTracking

        streamlines = LocalTracking(directionGetter,
                                    classifier,
                                    seed_points,
                                    mask_affine,
                                    step_size=cmdLineArgs.stepSize_,
                                    max_cross=cmdLineArgs.maxCross_,
                                    return_all=cmdLineArgs.returnAll_)

        # Compute streamlines and store as a list.
        streamlines = list(streamlines)

        # the length to trim is 5. but this can be changed
        # TODO make this user option
        from dipy.tracking.metrics import length
        streamlines = [s for s in streamlines if length(s) > cmdLineArgs.lenThresh_]

    elif cmdLineArgs.dirGttr_ == 'eudx':

        from dipy.tracking.eudx import EuDX

        eu = EuDX(FA.astype('f8'),
                  peakDirections,
                  seeds=seed_points,
                  odf_vertices=sphere.vertices,
                  a_low=cmdLineArgs.faClassify_,
                  step_sz=cmdLineArgs.stepSize_,
                  affine=mask_affine,
                  )

        # Compute streamlines and store as a list.
        streamlines = list(eu)

        from dipy.tracking.metrics import length
        streamlines = [s for s in eu if length(s) > cmdLineArgs.lenThresh_]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ TRK IO.TODO--> change it ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tracks_outputname = None

    if (cmdLineArgs.tractModel_ == 'csa') or (cmdLineArgs.tractModel_ == 'csd'):

        tracks_outputname = ''.join([cmdLineArgs.output_, '_',
                                     cmdLineArgs.dirGttr_,
                                     '_',
                                     cmdLineArgs.tractModel_,
                                     '_sh',
                                     str(cmdLineArgs.shOrder_),
                                     '.trk'])
    else:
        tracks_outputname = ''.join([cmdLineArgs.output_, '_',
                                     cmdLineArgs.dirGttr_,
                                     '_',
                                     cmdLineArgs.tractModel_,
                                     '.trk'])

    print('The output tracks name is: {}'.format(tracks_outputname))

    from dipy.io.trackvis import save_trk
    save_trk(tracks_outputname, streamlines, mask_affine, maskData.shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CONNECTIVITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # the number of streamlines, we will use later to normalize
    numStreams = len(streamlines)
    print('THE NUMBER OF STREAMS IS {0}'.format(numStreams))
    sys.stdout.flush()

    if cmdLineArgs.fsSegs_:

        for i in xrange(len(cmdLineArgs.fsSegs_)):

            print('\n\nnow making the connectivity matrices for: '
                  '{}'.format(str(cmdLineArgs.fsSegs_[i])))
            sys.stdout.flush()

            fsSegs_img = nib.load(cmdLineArgs.fsSegs_[i])
            fsSegs_data = fsSegs_img.get_data().astype(np.int16)

            # lets get the name of the seg to use
            # in the output writing
            segBaseName = ntpath.basename(
                ntpath.splitext(ntpath.splitext(cmdLineArgs.fsSegs_[i])[0])[0])

            from dipy.tracking.utils import connectivity_matrix
            M, grouping = connectivity_matrix(streamlines, fsSegs_data, affine=mask_affine,
                                              return_mapping=True,
                                              mapping_as_streamlines=True)
            # get rid of the first row yo...
            # because these are connections to '0' which we do not want
            M[:1, :] = 0
            M[:, :1] = 0

            print('here is the connectivity')
            print (M)

            import csv

            with open(''.join([cmdLineArgs.output_, '_', segBaseName,
                               '_sl_count.csv']), "wb") as f:
                writer = csv.writer(f)
                writer.writerows(M)

            # lets also make matrix of the fiber lengths
            # get the size that this should be...

            fib_lengths = np.zeros(M.shape).astype(np.float32)
            fib_len_sd = np.zeros(M.shape).astype(np.float32)

            # by row
            for x in xrange(M.shape[0]):
                # by column
                for y in xrange(M.shape[1]):

                    # check if entry in dict, if so, do more stuff
                    if (x, y) in grouping:

                        from dipy.tracking.utils import length

                        # now we record these values
                        stream_group = length(grouping[x, y], affine=mask_affine)
                        fib_lengths[x, y] = np.around(np.nanmean(list(stream_group)), 2)

                        stream_group = length(grouping[x, y], affine=mask_affine)
                        fib_len_sd[x, y] = np.around(np.nanstd(list(stream_group)), 2)

                        # if streamROIs, let's save each density image
                        if cmdLineArgs.streamROI_:

                            # dont measure to outside label, which is 0
                            if x == 0:
                                continue

                            print('saving ROI for each streamline')
                            sys.stdout.flush()

                            from dipy.tracking.utils import density_map

                            # just get the image dims we want
                            dimX = maskData.shape[0]
                            dimY = maskData.shape[1]
                            dimZ = maskData.shape[2]

                            # extract the density data
                            streamROIDens = density_map(streamlines=grouping[x, y],
                                                        vol_dims=(dimX, dimY, dimZ),
                                                        affine=mask_affine)

                            # format the ouptu name... label with same number as seg ROI number
                            # will output segs in 'upper triangle' of square mat
                            outName = ''.join([cmdLineArgs.output_,
                                               '_streamROI',
                                               '_x', str(x),
                                               '_y', str(y),
                                               '_dens.nii.gz'])

                            # make data into nifit image
                            streamROIDensImage = nib.Nifti1Image(streamROIDens.astype(np.float32), mask_affine)

                            # put the final bowtie on it
                            nib.save(streamROIDensImage, outName)

            # save the files
            with open(''.join([cmdLineArgs.output_, '_', segBaseName, '_sl_avglen.csv']), "wb") as f:
                writer = csv.writer(f)
                writer.writerows(fib_lengths.astype(np.float32))
            # and also save as npz
            np.savez_compressed(''.join([cmdLineArgs.output_, '_', segBaseName, '_sl_avglen.npz']),
                                fib_lengths.astype(np.float32))

            with open(''.join([cmdLineArgs.output_, '_', segBaseName, '_sl_stdlen.csv']), "wb") as f:
                writer = csv.writer(f)
                writer.writerows(fib_len_sd.astype(np.float32))
            # npz
            np.savez_compressed(''.join([cmdLineArgs.output_, '_', segBaseName, '_sl_stdlen.npz']),
                                fib_len_sd.astype(np.float32))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ StreamROIsTarg ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if cmdLineArgs.ROIsTarg_:

        for idx_ROIsTarg in xrange(len(cmdLineArgs.ROIsTarg_)):

            print('\n\nnow making the stream ROI from targ for: '
                  '{}'.format(str(cmdLineArgs.ROIsTarg_[idx_ROIsTarg])))
            sys.stdout.flush()

            targ_img = nib.load(cmdLineArgs.ROIsTarg_[idx_ROIsTarg])
            targ_data = targ_img.get_data().astype(np.int16)

            # import the function that we need
            from dipy.tracking.utils import target

            # get unique labels fo the targ image
            labelVals = np.unique(targ_data)
            # get rid of label 0
            labelVals = labelVals[1:]

            # 1 density
            dimX = maskData.shape[0]
            dimY = maskData.shape[1]
            dimZ = maskData.shape[2]

            # preallocate array for storing densities
            if cmdLineArgs.ROIsTargMult_:
                densitiesMat = np.zeros((dimX, dimY, dimZ, len(labelVals)))

            # now loop over these unique values
            for idx_labVals in xrange(len(labelVals)):

                labValue = labelVals[idx_labVals]

                if labValue == 0:
                    continue

                print('current targ = {}'.format(str(labValue)))

                # get just that area in the label image
                maskROI = (targ_data == labValue)

                targROI = target(streamlines, maskROI,
                                 affine=mask_affine,
                                 include=True)

                # need to list it, because target returns a generator
                targROI = list(targROI)

                # now we gonna save trk and density
                baseOutName = ntpath.basename(
                    ntpath.splitext(ntpath.splitext(cmdLineArgs.ROIsTarg_[idx_ROIsTarg])[0])[0])

                from dipy.tracking.utils import density_map
                # the dim{X,Y,Z} are still in scope
                streamROIDens = density_map(streamlines=targROI,
                                            vol_dims=(dimX, dimY, dimZ),
                                            affine=mask_affine)

                # make data into nifit image
                targROIImage = nib.Nifti1Image(streamROIDens.astype(np.float32), mask_affine)

                outName = ''.join([cmdLineArgs.output_, '_',
                                   baseOutName,
                                   '_targetROI',
                                   '_lab', str(labValue),
                                   '_dens.nii.gz'])

                # put the final bowtie on it
                nib.save(targROIImage, outName)

                # the actual trks
                outName = ''.join([cmdLineArgs.output_, '_',
                                   baseOutName,
                                   '_targetROI',
                                   '_lab', str(labValue),
                                   '.trk'])

                # already imported above
                # from dipy.io.trackvis import save_trk
                save_trk(outName, targROI, mask_affine, maskData.shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DENSITY MAP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from dipy.tracking.utils import density_map

    dimX = maskData.shape[0]
    dimY = maskData.shape[1]
    dimZ = maskData.shape[2]

    densityData = density_map(streamlines=streamlines,
                              vol_dims=(dimX, dimY, dimZ),
                              affine=mask_affine)
    # save the image
    print("saving the new image yo")
    sys.stdout.flush()

    density_image = nib.Nifti1Image(densityData.astype(np.float32), mask_affine)

    density_outputname = ''.join([cmdLineArgs.output_,
                                  '_',
                                  cmdLineArgs.dirGttr_,
                                  '_',
                                  cmdLineArgs.tractModel_,
                                  '_density.nii.gz'])

    nib.save(density_image, density_outputname)

if __name__ == '__main__':
    main()
