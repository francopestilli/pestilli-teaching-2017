import os
import sys

# for amico, just do it
import spams
import argparse
import numpy as np
import nibabel as nib
import h5py

__author__ = 'jfaskowitz'

"""

Josh Faskowitz
Indiana University

built on top of scripts written at
Imaging Genetics Center,
University of Southern California

inspiration taken from the dipy website

"""

np.seterr(all='ignore')

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


class readArgs:
    
    def __init__(self):

        # initialize all the variables to empty strings
        
        # required vars
        self.dwi_ = ''
        self.mask_ = ''
        self.bvec_ = ''
        self.bval_ = ''
        self.output_ = ''
        
        # other vars
        self.fa_resp_thr_ = 0.7
        self.sh_order_ = 6
        self.recursiveResp_ = False
        self.power_map_ = False
        self.amico_ = False
        self.fit_csd_ = False
        self.fit_csa_ = False
        self.fit_dki_ = False
        self.fit_sfm_ = False
        self.num_peaks_vol_ = False

        # peaks params
        self.peaks_sep_angle_ = 25
        self.peaks_rel_thr_ = 0.5

    @staticmethod
    def collect_args():

        argParseObj = argparse.ArgumentParser(description="General wrapper for getting maps "
                                                          "of various microstructure indicies")
        
        #
        # required args
        #
         
        # this is the whole dwi with all the volumes yo
        argParseObj.add_argument('-dwi', help="This is the path to dwi yo", type=str, nargs='?', required=True)

        # this is an easy binary mask that you should aready have
        argParseObj.add_argument('-mask', nargs='?', required=True,
                                 help="This is the path to mask yo", type=str)

        # this is the bvecs file, which should be eddy and flirt rotated
        argParseObj.add_argument('-bvec', nargs='?', required=True,
                                 help="", type=str)

        # this is the bval, which sometimes is the same for the whole dataset
        argParseObj.add_argument('-bval', nargs='?', required=True,
                                 help="", type=str)

        # the output should be provided, so we can name it
        argParseObj.add_argument('-output', nargs='?', required=True,
                                 help="Add the name of the output prefix you want", type=str)

        #
        # opts 
        #
        
        argParseObj.add_argument('-fa_response_thr', nargs='?', required=False,
                                 help="fa classifier for tissue clasifier", type=str)

        argParseObj.add_argument('-sh_order', nargs='?', required=False,
                                 help="sh order of the model yo", type=str)

        argParseObj.add_argument('-peak_separation_angle', nargs='?', required=False,
                                 help="sh order of the model yo", type=str)

        argParseObj.add_argument('-peak_relative_thresh', nargs='?', required=False,
                                 help="sh order of the model yo", type=str)
        # 
        # do micrstucture stuff
        #

        argParseObj.add_argument('-fit_csd', required=False,
                                 help="if you want to fit csd", action='store_true')

        argParseObj.add_argument('-fit_csa', required=False,
                                 help="if you want to fit csa", action='store_true')

        argParseObj.add_argument('-rescursive_response', required=False,
                                 help="if you want recursive reponse", action='store_true')

        argParseObj.add_argument('-make_power_map', required=False,
                                 help="if you want recursive reponse", action='store_true')

        argParseObj.add_argument('-do_amico', required=False,
                                 help="if you want amico noddi", action='store_true')

        argParseObj.add_argument('-fit_sfm', required=False,
                                 help="if you want fit sparse fascile model", action='store_true')

        argParseObj.add_argument('-num_peak_dirs_vol', required=False,
                                 help="if you output a peak dirs volume",
                                 action='store_true')

        return argParseObj

    def check_args(self, args_to_check):

        # this is a builtin function to parse the arguments of the arg parse module
        args = args_to_check.parse_args()
        print(args)

        # get the arguments and store them as variables yo
        self.dwi_ = args.dwi
        self.mask_ = args.mask
        self.bvec_ = args.bvec
        self.bval_ = args.bval
        self.output_ = args.output
        
        if args.fa_response_thr: 
            self.fa_resp_thr_ = args.fa_response_thr
        if args.sh_order:
            self.sh_order_ = args.sh_order
        if args.rescursive_response:
            self.recursiveResp_ = True
        if args.make_power_map:
            self.power_map_ = True
        if args.do_amico:
            self.amico_ = True
        if args.fit_csd:
            self.fit_csd_ = True
        if args.fit_csa:
            self.fit_csa_ = True
        if args.fit_sfm:
            self.fit_sfm_ = True
        if args.num_peak_dirs_vol:
            self.num_peaks_vol_ = True
        if args.peak_separation_angle:
            self.peaks_sep_angle_ = args.peak_separation_angle
        if args.peak_relative_thresh:
            self.peaks_rel_thr_ = args.peak_relative_thresh

        if not os.path.exists(self.dwi_):
            # not cool yo
            print('The dwi image does not exist. exiting')
            exit(1)
        if not os.path.exists(self.mask_):
            # not cool yo
            print('The mask image does not exist. exiting')
            exit(1)
        if not os.path.exists(self.bvec_):
            # not cool yo
            print('The bvec image does not exist. exiting')
            exit(1)
        if not os.path.exists(self.bval_):
            # not cool yo
            print('The bval image does not exist. exiting')
            exit(1)

        # optional arguments, first check if the string exists...
        # if string does not exist, dont worry about it
        if args.fa_response_thr and not isfloat(self.fa_response_thr):
            print("fa classifier needs to be a number yo")
            exit(1)
        if args.sh_order and not isint(self.sh_order_):
            print("sh needs to be an in yo... probs 2 or 4 or 6 or 8")
            exit(1)
        if args.peak_separation_angle and not isfloat(self.peaks_sep_angle_):
            print("separation angle needs to be float")
            exit(1)
        if args.peak_relative_thresh and not isfloat(self.peaks_rel_thr_):
            print("relative thresh angle needs to be float")
            exit(1)

        print('args look good dude')

def save_pam_h5(peaks_n_metrics, out_name):

    fod_coeff = peaks_n_metrics.shm_coeff

    # add other elements of csd_peaks to npz file
    fod_gfa = peaks_n_metrics.gfa
    fod_qa = peaks_n_metrics.qa
    fod_peak_dir = peaks_n_metrics.peak_dirs
    fod_peak_val = peaks_n_metrics.peak_values
    fod_peak_ind = peaks_n_metrics.peak_indices

    with h5py.File(out_name, 'w') as hf:
        group1 = hf.create_group('PAM')
        group1.create_dataset('coeff', data=fod_coeff, compression="gzip")
        group1.create_dataset('gfa', data=fod_gfa, compression="gzip")
        group1.create_dataset('qa', data=fod_qa, compression="gzip")
        group1.create_dataset('peak_dir', data=fod_peak_dir, compression="gzip")
        group1.create_dataset('peak_val', data=fod_peak_val, compression="gzip")
        group1.create_dataset('peak_ind', data=fod_peak_ind, compression="gzip")

def write_num_peaks(peaks, affine, out_name):
    # also print out a volume for number peaks
    # sum the values greater than 5 in the third dimension
    img = nib.Nifti1Image(np.sum(peaks.peak_values > 0, 3), affine)
    nib.save(img, out_name)

def main():
    
    #
    # get the command line stuff
    #
    
    # initialize obj
    params = readArgs()
    # read in from the command line
    argParseObj = params.collect_args()
    # make sure args look nice
    params.check_args(argParseObj)

    # flush python stuff to the screen
    sys.stdout.flush()

    # ~~~~~~~~~~ Read in images ~~~~~~~~~~~~~~~~~~~~~~  

    # convert the strings into images we can do stuff with in dipy yo!
    dwiImage = nib.load(params.dwi_)
    maskImage = nib.load(params.mask_)

    # ~~~~~~~~~~ Bvals and Bvecs stuff ~~~~~~~~~~~~~~~

    # read the bvals and bvecs into data artictecture that dipy understands
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(params.bval_, 
                                    params.bvec_)

    # need to create the gradient table yo
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs, b0_threshold=100)

    # get the data from image objects
    dwiData = dwiImage.get_data()
    maskData = maskImage.get_data()

    maskAffine = maskImage.get_affine()
    dwiAffine = dwiImage.get_affine()

    print('getting sphere')
    from dipy.data import get_sphere
    sphere = get_sphere('repulsion724')

    # mask the dwiData by mask provided
    from dipy.segment.mask import applymask
    # apply the provided mask to the dwi data
    dwiData = applymask(dwiData, maskData)

    # output some stuff to screen
    print('dwiData.shape (%d, %d, %d, %d)' % dwiData.shape)
    print('\nYour bvecs look like this:{0}'.format(bvecs))
    print('\nYour bvals look like this:{0}\n'.format(bvals))
    sys.stdout.flush()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RESPONSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # after this step we will always have info for response func

    if params.recursiveResp_:

        outName = params.output_ + 'recurResponse' + str(params.sh_order_) + '.npz'

        if os.path.isfile(outName):
            print("already found the recursive stuffs, will load it up")
            sys.stdout.flush()

            responseTemp = np.load(outName)
            response = responseTemp['arr_0']
            response = response.item()

        else:

            print("using dipy function recursive_response to generate response")
            sys.stdout.flush()

            from dipy.reconst.csdeconv import recursive_response

            import dipy.reconst.dti as dti
            # least weighted is standard
            tenmodel = dti.TensorModel(gtab)

            print("fitting tensor")
            tenfit = tenmodel.fit(dwiData, mask=maskData)
            print("done fitting tensor")
            sys.stdout.flush()

            from dipy.reconst.dti import fractional_anisotropy
            FA = fractional_anisotropy(tenfit.evals)
            MD = dti.mean_diffusivity(tenfit.evals)
            wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

            print("doing recusive stuffs...")
            sys.stdout.flush()
            response = recursive_response(gtab, dwiData, mask=wm_mask, 
                                          sh_order=int(params.sh_order_),
                                          peak_thr=0.01, 
                                          init_fa=0.08,
                                          init_trace=0.0021, 
                                          iter=8, 
                                          convergence=0.001,
                                          parallel=True,
                                          nbr_processes=4)
            print("finish doing recusive stuffs")
            sys.stdout.flush()

            # lets save the response as an np object to the disk
            np.savez_compressed(outName, response, 'arr_0')

    # if we not fitting recursively...
    # if we fitting something that needs response...
    elif params.fit_csd_ or params.fit_sfm_:
        
        print("using dipy function auto_response to generate response")
        sys.stdout.flush()

        from dipy.reconst.csdeconv import auto_response
        response, ratio = auto_response(gtab, 
                                        dwiData, 
                                        roi_radius=10, 
                                        fa_thr=float(params.fa_resp_thr_))

        print("making the response  with sh of {}\n".format(params.sh_order_))
        sys.stdout.flush()

        # lets save the response as an np object to the disk
        # TODO fix this naming...because the autoresponse doesnt use sh...
        # make sure it works with stuff
        outName = params.output_ + 'autoResponse.npz'
        np.savez_compressed(outName, response, 'arr_0')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if params.fit_csd_:
        print("fitting csd")
        sys.stdout.flush()

        from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=int(params.sh_order_))

        print("making peaks from model")
        sys.stdout.flush()

        from dipy.direction import peaks_from_model
        csd_peaks = peaks_from_model(model=csd_model,
                                     data=dwiData,
                                     sphere=sphere,
                                     relative_peak_threshold=float(params.peaks_rel_thr_),
                                     min_separation_angle=float(params.peaks_sep_angle_),
                                     mask=maskData,
                                     return_sh=True,
                                     parallel=True,
                                     nbr_processes=4)
        print("done making peaks from model")
        sys.stdout.flush()

        print('writing to the file the coefficients for sh order of: {0}'.format(str(params.sh_order_)))
        # lets write this to the disk you
        # fullOutput = outputname + '_csdCoeffs_params.sh_order__' + str(params.sh_order_) + '.npz'
        # np.savez_compressed(fullOutput,fod_coeff,'arr_0')
        fullOutput = params.output_ + '_csdCompute_sh' + str(params.sh_order_) + '.h5'
        save_pam_h5(csd_peaks, fullOutput)

        # =======================================================================

        # lets also write out a gfa image yo, just for fun
        gfa_img = nib.Nifti1Image(csd_peaks.gfa.astype(np.float32), dwiImage.get_affine())

        # make the output name yo
        gfaOutputName = ''.join([params.output_, '_gfa_csd_sh', str(params.sh_order_), '.nii.gz'])
        print('The output FA name is: {}'.format(gfaOutputName))

        # same this FA
        nib.save(gfa_img, gfaOutputName)

        if params.num_peaks_vol_:
            outName = ''.join([params.output_, '_csa_numPeaks_sh', str(params.sh_order_), '.nii.gz'])
            write_num_peaks(csd_peaks, dwiImage.get_affine(), outName)

    if params.fit_csa_:
        print("fitting csa")
        sys.stdout.flush()

        from dipy.reconst.shm import CsaOdfModel, normalize_data
        csa_model = CsaOdfModel(gtab, int(params.sh_order_))

        print("making peaks from model")
        sys.stdout.flush()

        from dipy.direction import peaks_from_model
        csa_peaks = peaks_from_model(model=csa_model,
                                     data=dwiData,
                                     sphere=sphere,
                                     relative_peak_threshold=float(params.peaks_rel_thr_),
                                     min_separation_angle=float(params.peaks_sep_angle_),
                                     mask=maskData,
                                     return_sh=True,
                                     parallel=True,
                                     nbr_processes=4,
                                     normalize_peaks=True)

        print('writing to the file the coefficients for sh order of: {0}'.format(str(params.sh_order_)))
        # lets write this to the disk you
        # fullOutput = outputname + '_csdCoeffs_params.sh_order__' + str(params.sh_order_) + '.npz'
        # np.savez_compressed(fullOutput,fod_coeff,'arr_0')
        fullOutput = params.output_ + '_csaCompute_sh' + str(params.sh_order_) + '.h5'
        save_pam_h5(csa_peaks, fullOutput)

        # =======================================================================

        # lets also write out a gfa image yo, just for fun
        gfa_img = nib.Nifti1Image(csa_peaks.gfa.astype(np.float32), dwiImage.get_affine())

        # make the output name yo
        gfaOutputName = ''.join([params.output_, '_gfa_csa_sh', str(params.sh_order_), '.nii.gz'])
        print('The output FA name is: {}'.format(gfaOutputName))

        # same this FA
        nib.save(gfa_img, gfaOutputName)

        if params.num_peaks_vol_:
            outName = ''.join([params.output_, '_csd_numPeaks_sh', str(params.sh_order_), '.nii.gz'])
            write_num_peaks(csa_peaks, dwiImage.get_affine(), outName)

    if params.power_map_:
        print("going to also make anisotropic power map")
        sys.stdout.flush()

        from dipy.reconst.shm import anisotropic_power

        '''
        powerMap = anisotropic_power(csd_peaks.shm_coeff,
                                     norm_factor=0.00001,
                                     power=2,
                                     non_negative=True)

        powerMap_img = nib.Nifti1Image(powerMap.astype(np.float32), dwiImage.get_affine())
        # make output name
        pmOutputName = ''.join([params.output_, '_powerMapShCSD_', str(params.sh_order_), '.nii.gz'])
        nib.save(powerMap_img, pmOutputName)
        '''

        # copying from
        # https://gist.github.com/mpaquette/c785bb4d584297f1d8112c0725f69fbe

        print("going to make power map from signal too")
        from dipy.reconst.shm import sph_harm_lookup, smooth_pinv, normalize_data
        from dipy.core.sphere import HemiSphere

        smooth = 0.0

        normed_data = normalize_data(dwiData, gtab.b0s_mask)
        normed_data = normed_data[..., np.where(1 - gtab.b0s_mask)[0]]

        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        gtab2 = gradient_table_from_bvals_bvecs(gtab.bvals[np.where(1 - gtab.b0s_mask)[0]],
                                                gtab.bvecs[np.where(1 - gtab.b0s_mask)[0]])

        signal_native_pts = HemiSphere(xyz=gtab2.bvecs)

        sph_harm_basis = sph_harm_lookup.get(None)
        Ba, m, n = sph_harm_basis(int(params.sh_order_),
                                  signal_native_pts.theta,
                                  signal_native_pts.phi)

        L = -n * (n + 1)
        invB = smooth_pinv(Ba, np.sqrt(smooth) * L)

        # fit SH basis to DWI signal
        normed_data_sh = np.dot(normed_data, invB.T)

        powerMap = anisotropic_power(normed_data_sh,
                                     norm_factor=0.00001,
                                     power=2,
                                     non_negative=True)

        powerMap_img = nib.Nifti1Image(powerMap.astype(np.float32), dwiImage.get_affine())
        # make output name
        pmOutputName = ''.join([params.output_, '_powerMapShNative_', str(params.sh_order_), '.nii.gz'])
        nib.save(powerMap_img, pmOutputName)

    if params.amico_:

        # lets try to run the amico
        import amico

        ae = amico.Evaluation("run_amico", "subj")

        # convert bvecs to scheme
        amico.util.fsl2scheme(str(params.bval_), str(params.bvec_), 
                              schemeFilename='subj_amico.scheme')

        # full scheme path
        schemeText = ''.join([os.getcwd(), '/subj_amico.scheme'])
        ae.load_data(dwi_filename=params.dwi_,
                     scheme_filename=params.dwi_,
                     mask_filename=params.mask_, 
                     b0_thr=50)

        print("start: setting amico model and generating kernel")
        sys.stdout.flush()

        ae.set_model("NODDI")
        ae.generate_kernels()

        print("finish: setting amico model and generating kernel")
        sys.stdout.flush()

        ae.load_kernels()

        print("start: fitting amico")
        sys.stdout.flush()

        ae.fit()

        print("finish: fitting amico")
        sys.stdout.flush()

        ae.save_results()

    if params.fit_dki_:
        print("going to fit DKI model")
        sys.stdout.flush()

        import dipy.reconst.dki as dki

        dkiModel = dki.DiffusionKurtosisModel(gtab)
        dkifit = dkiModel.fit(dwiData, maskData)

        # TODO WIP

    if params.fit_sfm_:
        print("fitting sparse fasicle model")
        sys.stdout.flush()

        # run example from website
        import dipy.reconst.sfm as sfm
        sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                           l1_ratio=0.5, alpha=0.001,
                                           response=response[0])
        # sf_fit = sf_model.fit(dwiData)

        from dipy.direction import peaks_from_model
        sf_peaks = peaks_from_model(model=sf_model,
                                    data=dwiData,
                                    sphere=sphere,
                                    mask=maskData,
                                    relative_peak_threshold=float(params.peaks_rel_thr_),
                                    min_separation_angle=float(params.peaks_sep_angle_),
                                    return_sh=False,
                                    parallel=True,
                                    nbr_processes=4)

        '''
        # lets also write out a gfa image yo, just for fun
        gfa_img = nib.Nifti1Image(sf_peaks.gfa.astype(np.float32), dwiImage.get_affine())

        # make the output name yo
        gfaOutputName = ''.join([params.output_, '_sfm_gfa_sh', str(params.sh_order_), '.nii.gz'])
        print('The output name is: {}'.format(gfaOutputName))

        # same this FA
        nib.save(gfa_img, gfaOutputName)

        print('writing to the file the coefficients for sh order of: {0}'.format(str(params.sh_order_)))
        # lets write this to the disk you
        # fullOutput = outputname + '_csdCoeffs_params.sh_order__' + str(params.sh_order_) + '.npz'
        # np.savez_compressed(fullOutput,fod_coeff,'arr_0')
        fullOutput = params.output_ + '_sfmCompute_sh' + str(params.sh_order_) + '.h5'
        save_pam_h5(sf_peaks, fullOutput)
        '''

        if params.num_peaks_vol_:
            outName = ''.join([params.output_, '_sfm_numPeaks.nii.gz'])
            write_num_peaks(sf_peaks, dwiImage.get_affine(), outName)

if __name__ == '__main__':
    main()
