# PixelPred2Seg

This repo provides sample instructions on how to generate segmentation for EM images from pixelwise predictions (right now only for membrane prediction). *Important Note: these instructions are merely a guideline, or perhaps, a starting point for producing segmentation from pixel prediction. It is expected that the methods, codes, parameters needs to be changed for optimal performance for the pixelwise prediction the user is generating.*

There are two stages of this procedure: first generate an over-segmentation and then merge the over-segmented regions using an agglomeration appraoch. Finally, this repo also provides python scripts to compare two segmentations by Variation of Information (VI) and Rand Error (RE).

# Oversegmentation

Given pixelwise membrane predictions, the generate_watershed_RD.py python script will generate the watershed, where R is the mode of operation. For R=2, it will generate watershed for each sections separately; for R=3, it will run a 3D watershed. Please change the filenames/paths to match your locations.

One may need to modify/smooth the predictions and perhaps apply operations such as distance transform on pixelwise predictions to compute the desired over-segmentation. The find_seeds_sa.py provides some hints for performing such operation. It is important to remember that, ideally the pixelwise predictions should be able to generate the optimal over-segemntation (e.g., there is no false merge and the #regions is within 5 times of the actual neuron processes) just by thresholding. Therefore, this is what I suggest the users to achieve.


# Agglomeration

I recommend using the Neuroproof from the github repo: https://github.com/paragt/Neuroproof_minimal. The instructions on how to use it is provided in the repo, please also read the PLoS ONE paper Parag, T. et. al (2015). A Context-Aware Delayed Agglomeration Framework for Electron Microscopy Segmentation (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0125825) for a clear understanding of the process. I am restating them here for the filenames used in the geenrate_watershed_2D or 3D.py. 

To learn the superpixel boundary classifier: 

~/work/Neuroproof_minimal/Neuroproof_minimal/build/NeuroProof_stack_learn -watershed /home/paragt/work/pipeline_codes/train_vol/seg.h5  stack -prediction /home/paragt/work/pipeline_codes/train_vol/pixel_pred.h5  stack -classifier /home/paragt/work/pipeline_codes/train_vol/agglo_classifier_itr1.h5  -groundtruth /home/paragt/work/pipeline_codes/train_vol/ecs_seg_groundtruth_cropped.h5  stack -strategy 2  -iteration 1 -nomito 

To run agglomeration :

~/work/Neuroproof_minimal/Neuroproof_minimal/build/NeuroProof_stack -watershed /home/paragt/work/pipeline_codes/train_vol/seg.h5  stack -prediction /home/paragt/work/pipeline_codes/train_vol/pixel_pred.h5  stack -classifier /home/paragt/work/pipeline_codes/train_vol/agglo_classifier_itr1.h5  -output /home/paragt/work/pipeline_codes/train_vol/tst_seg.h5 stack -algorithm 1  -threshold 0.1 -min_region_sz 0 -nomito

# Comparison

The comparison code is a snippet from the GALA github repo (https://github.com/janelia-flyem/gala/tree/master/gala) that has been used for multiple publications and for the EM segmentation challenge CREMI (https://cremi.org/). The parameters that have been faithfully computing the segmentation error for me are the provided below, I suggest the users to use the same values so that we can make a fair comparison among all variants. 

 ./comparestacks --stack1 result_filename.h5  --stackbase seg_groundtruth_filename.h5 --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100


# Build

The libraries used for the github repo: https://github.com/paragt/Neuroproof_minimal should also create the environment for running the watershed algo. I am providing the instructions again.

Linux: Install miniconda on your workstation. Create and activate the conda environment using the following commands:

  conda create -n my_conda_env -c flyem vigra=1.10 opencv 

  source activate my_conda_env
  
  conda install scipy
  
  conda install scikit-learn

Then follow the usual procedure of building Neuroproof_minimal:

  mkdir build
 
  cd build

  cmake -DCMAKE_PREFIX_PATH=[CONDA_ENV_PATH]/my_conda_env ..

  make
  
  
  
# Helpful hints for finding markers for watershed

Provided by Lee Kamentsky:

In terms of parameters for finding seeds:
sigma_xy and sigma_z are the best controls over the oversegmentation if one uses the smoothing method - higher values perform an ersatz distance transform by smoothing the membranes into the cytoplasm. If you find yourself using high values, maybe it's time to switch to the distance transform.

The distance transform uses a lot of memory, especially if is is run on a desktop. Large distances don't matter, so you can run the distance transform on blocks, with the amount of overlap being the largest distance you really care about. Most of the parameters relate to the parallelization of the distance transform.

The user may want to apply a bit of smoothing prior to the watershed. One can just run scipy.ndimage.gaussian_filter prior to the watershed to do that, e.g.: smoothed = gaussian_filter(prob, (self.sigma_z, self.sigma_xy, self.sigma_xy))
