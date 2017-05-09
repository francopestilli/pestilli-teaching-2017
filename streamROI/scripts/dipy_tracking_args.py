import os
import argparse

__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University
University of Southern California

inspiration taken from the dipy website

"""

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        print("is not float")
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        print("is not int")
        return False


class GetParams:
    def __init__(self):

        # initialize all the variables to empty strings

        # the basics
        self.dwi_ = ''
        self.mask_ = ''
        self.bvec_ = ''
        self.bval_ = ''
        self.output_ = 'output'

        self.tractModel_ = ''
        self.dirGttr_ = ''

        # for generating the connectivity matrix
        self.fsSegs_ = ''

        # for seeding / seeds
        self.wmMask_ = ''
        self.actClasses_ = []
        self.seedDensity_ = 1
        self.randSeed_ = False
        self.limitTotSeeds_ = 0
        self.saveSeedPoints_ = False
        self.seedPointsFile_ = ''

        # for modeling the diffusion / fibers
        self.shOrder_ = 6
        self.coeffFile_ = ''
        self.maxCross_ = None
        self.returnAll_ = False

        # if you using fa classifier
        self.faClassify_ = 0.2
        self.lenThresh_ = 5
        self.stepSize_ = 0.5
        self.sphere_ = 'symmetric724'

        # extra ishes
        self.doLife_ = False
        self.streamROI_ = False
        self.ROIsTarg_ = ''
        self.ROIsTargMult_ = ''

    def do_work(self):

        argParse = argparse.ArgumentParser(description="Dipy wrapper written by Josh Faskowitz.")

        # this is the whole dwi with all the volumes yo
        argParse.add_argument('-dwi',
                              help="(Required) This is the path to dwi nifti",
                              type=str,
                              nargs=1,
                              required=True)

        # this is an easy binary mask that you should aready have
        argParse.add_argument('-mask',
                              help="(Required) This is the path to the binary mask nifti",
                              type=str,
                              nargs=1,
                              required=True)

        # this is the bvecs file, which should be eddy and flirt rotated
        argParse.add_argument('-bvec',
                              help="(Required) The bvecs file needs to be normalized "
                                   "(unit vector) for dipy to work, fsl format",
                              type=str,
                              nargs=1,
                              required=True)

        # this is the bval, which sometimes is the same for the whole dataset
        argParse.add_argument('-bval',
                              help="(Required) Standard bvals, fsl format",
                              type=str,
                              nargs=1,
                              required=True)

        # the output should be provided, but I guess it could be optional
        argParse.add_argument('-output',
                              help="Add the name of the output prefix you want. Default is: ouput ",
                              type=str,
                              nargs=1,
                              required=False)

        # connectivity matrix stuff

        # freesurfer segs
        argParse.add_argument('-segs',
                              help="Parcellation in same space as dwi, can be multiple segs",
                              type=str,
                              nargs='*',
                              required=False)

        # tractography model (engine) baby!
        argParse.add_argument('-tract_model',
                              help="tracotraphy model you want to use: [csa, csd, dti]",
                              choices=['csa', 'csd', 'dti'],
                              nargs=1,
                              type=str,
                              required=True)

        argParse.add_argument('-direction_gttr',
                              help="direction getter for dipy, [deterministic,probabilisitc,eudx]",
                              choices=['deterministic', 'probabilistic', 'eudx'],
                              nargs=1,
                              type=str,
                              required=False)

        # seeding stuffs
        # self.wmMask_ = ''
        # self.actClasses_ = []
        # self.seedDensity_ = 1
        # self.randSeed_ = 0
        # self.limitTotSeeds_ = 0

        # white matter mask used to seed the tractography
        argParse.add_argument('-wm_mask',
                              help="White matter mask, seeds will be in this tractography "
                                   "(otherwise will use wholebrain mask provided previously",
                              type=str,
                              nargs=1,
                              required=False)

        argParse.add_argument('-act_tissues',
                              help="Tissues classifiers to use with ACT (generated from FSL FAST). csf, gm, wm,",
                              type=str,
                              nargs=3,
                              required=False)

        argParse.add_argument('-seed_density',
                              help="Density you want to seed at per voxel",
                              type=int,
                              nargs=1,
                              required=False)

        argParse.add_argument('-rand_seed',
                              help="random seed flag. default: random location for seed within each voxel",
                              action='store_true',
                              required=False)

        argParse.add_argument('-tot_seeds',
                              help="limit the total number of seeds.",
                              nargs=1,
                              type=int,
                              required=False)

        argParse.add_argument('-save_seed_points',
                              help='save seed points as npz for fuutre use. use this flag for true',
                              action='store_true',
                              required=False)

        argParse.add_argument('-seed_points',
                              help='seed points npz to be used. realize that this should only be used on same '
                                   'subject+affine position as these seeds will be in space of orig seed generation',
                              nargs=1,
                              type=str,
                              required=False)

        # for modeling the diffusion / fibers
        # self.shOrder_ = 6
        # self.coeffFile_ = ''
        # self.maxCross_ = None
        # self.returnAll_ = False

        argParse.add_argument('-sh_order',
                              help="Spherical haromic order of the model",
                              nargs=1,
                              type=int,
                              required=False)

        argParse.add_argument('-coeff_file',
                              help="if you already generated coeffs",
                              nargs=1,
                              type=str,
                              required=False)

        argParse.add_argument('-seed_cross',
                              help="number of crossings to allow (maximum) when initializing seeds. default = None",
                              nargs=1,
                              type=int,
                              required=False)

        argParse.add_argument('-return_all',
                              help="return all paramters in local tracking, flag. not recommended",
                              action='store_true',
                              required=False)

        # extras

        argParse.add_argument('-fa_thresh',
                              help="fa classifier for tissue clasifier",
                              nargs=1,
                              type=float,
                              required=False)

        argParse.add_argument('-length_thresh',
                              help='length thrshold to cut off streamlines. default = 5mm',
                              nargs=1,
                              type=int,
                              required=False)

        argParse.add_argument('-step_size',
                              help="step size of streamline trackin",
                              nargs=1,
                              type=float,
                              required=False)

        argParse.add_argument('-sphere',
                              help="chose sphere you want from default dipy options. default: symmetric724 ",
                              nargs=1,
                              choices=['symmetric362', 'symmetric642', 'symmetric724', 'repulsion724', 'repulsion100'],
                              type=str,
                              required=False)

        argParse.add_argument('-do_LiFE',
                              help='linear fascile evaluation',
                              action='store_true',
                              required=False)

        argParse.add_argument('-streamROIs',
                              help='save density ROI for tracts connecting one ROI to another',
                              action='store_true',
                              required=False)

        argParse.add_argument('-ROIsTarg',
                              help='targ labels for streams to pass through',
                              type=str,
                              nargs='*',
                              required=False)

        argParse.add_argument('-ROIsTarg_multi',
                              help='targ labels multiplier for rdm making',
                              type=str,
                              nargs='*',
                              required=False)

        ###########################################################################
        ###########################################################################
        ###########################################################################

        # this is a builtin function to parse the arguments of the arg parse module
        args = argParse.parse_args()

        print("here are the args read {}".format(args))

        # get the arguments and store them as variables yo
        self.dwi_ = args.dwi[0]
        if not os.path.exists(self.dwi_):
            # not cool yo
            print('The dwi image does not exist. exiting')
            exit(1)

        self.mask_ = args.mask[0]
        if not os.path.exists(self.mask_):
            # not cool yo
            print('The mask image does not exist. exiting')
            exit(1)

        self.bvec_ = args.bvec[0]
        if not os.path.exists(self.bvec_):
            # not cool yo
            print('The bvec image does not exist. exiting')
            exit(1)

        self.bval_ = args.bval[0]
        if not os.path.exists(self.bval_):
            # not cool yo
            print('The bval image does not exist. exiting')
            exit(1)

        if args.output:
            self.output_ = args.output[0]

        if args.segs:
            self.fsSegs_ = args.segs
        if self.fsSegs_ :
            for x in range(len(args.segs)):
                if not os.path.exists(self.fsSegs_[x]):
                    # not cool yo
                    print("path for segs does not exist yo:\n{}".format(self.fsSegs_[x]))
                    print('The fs Segs image does not exist. exiting')
                    exit(1)

        self.tractModel_ = args.tract_model[0]

        if args.direction_gttr:
            self.dirGttr_ = args.direction_gttr[0]

        if args.wm_mask:
            self.wmMask_ = args.wm_mask[0]
        if self.wmMask_ and not os.path.exists(self.wmMask_):
            # not cool yo
            print('The fs wm image does not exist. exiting')
            exit(1)

        if args.act_tissues:
            self.actClasses_ = args.act_tissues
        if self.actClasses_:
            for x in range(0, 3):
                if not os.path.exists(self.actClasses_[x]):
                    print("path for act tissue does not exist yo:\n{}".format(self.actClasses_[x]))
                    exit(1)

        if args.seed_density:
            self.seedDensity_ = args.seed_density[0]
        if self.seedDensity_ and not isint(self.seedDensity_) or int(self.seedDensity_) > 25:
            print("seed density not legit yo. you dont want density > 25. ttooooo big")
            print("seed density: {}".format(self.seedDensity_))
            exit(1)

        if args.rand_seed:
            self.randSeed_ = args.rand_seed

        if args.tot_seeds:
            self.limitTotSeeds_ = args.tot_seeds[0]
        if args.tot_seeds and not isfloat(self.limitTotSeeds_):
            print("need to provide int that limits seeds yo")
            exit(1)

        if args.save_seed_points:
            self.saveSeedPoints_ = args.save_seed_points

        if args.seed_points:
            self.seedPointsFile_ = args.seed_points
        if args.seed_points and not os.path.exists(self.seedPointsFile_[0]):
            print('seed points npz file invalid yo')
            exit(1)

        if args.sh_order:
            self.shOrder_ = args.sh_order[0]
        if args.sh_order and not isint(self.shOrder_):
            print("sh needs to be an in yo... probs 2 or 4 or 6 or 8")
            exit(1)

        if args.coeff_file:
            self.coeffFile_ = args.coeff_file[0]
        if self.coeffFile_ and not os.path.exists(self.coeffFile_):
            print ("coeff file does not exist. exiting")
            exit(1)

        if args.seed_cross:
            self.maxCross_ = int(args.seed_cross[0])
        if args.seed_cross and not isint(self.maxCross_):
            print("seed cross needs to be int")
            exit(1)

        if args.return_all:
            self.returnAll_ = args.return_all

        if args.fa_thresh:
            self.faClassify_ = args.fa_thresh[0]
        if args.fa_thresh and not isfloat(self.faClassify_):
            print("fa classifier needs to be a number yo")
            exit(1)

        if args.length_thresh:
            self.lenThresh_ = args.length_thresh[0]
        if args.length_thresh and not isint(self.lenThresh_):
            print('gotta put an int for lenth thresh yo')
            exit(1)

        if args.step_size:
            self.stepSize_ = args.step_size[0]
        if args.step_size and not isfloat(self.stepSize_):
            print('step size not valid yo')
            exit(1)

        if args.sphere:
            self.sphere_ = args.sphere[0]
        # don't need to stupid check this one because its restricted to choices anyways

        if args.do_LiFE:
            self.doLife_ = True

        if args.streamROIs:
            if len(args.segs) == 0:
                print('must also provide segs to do streamROIs')
                exit(1)
            self.streamROI_ = True

        if args.ROIsTarg:
            self.ROIsTarg_ = args.ROIsTarg
        if self.ROIsTarg_:
            for x in range(len(args.ROIsTarg)):
                if not os.path.exists(self.ROIsTarg_[x]):
                    # not cool yo
                    print("path for roiTarg exist yo:\n{}".format(self.ROIsTarg_[x]))
                    exit(1)

        # ROIsTarg_multi
        if args.ROIsTarg_multi:
            self.ROIsTargMult_ = args.ROIsTarg_multi
        if self.ROIsTargMult_:
            for x in range(len(args.ROIsTarg_multi)):
                if not os.path.exists(self.ROIsTargMult_[x]):
                    # not cool yo
                    print("path for roiTarg exist yo:\n{}".format(self.ROIsTargMult_[x]))
                    exit(1)

        print('args look good dude')

        # lets print the params to a file
        f = open('parsed_params.txt', 'w')
        f.write(str(args))
        f.close()

        return 0
