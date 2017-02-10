#import enum
#import luigi
import numpy as np
#import rh_logger
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from scipy.ndimage import grey_dilation, grey_erosion

##from ..algorithms.morphology import parallel_distance_transform
##from ..parameters import VolumeParameter, DatasetLocationParameter
##from ..targets.factory import TargetFactory
##from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin


##class SeedsMethodEnum(enum.Enum):
    ##'''Enumeration of the seed finding algorithms'''

    ##'''Find seeds by smoothing with a Gaussian and thresholding
    
    ##The probabilities are smoothed with an anisotropic Gaussian with
    ##different x/y and z sigmas, then the result is thresholded and
    ##connected components are found.
    ##'''
    ##Smoothing=1
    
    ##'''Find seeds by finding maxima in the distance transform
    
    ##The probabilities are thresholded, then the distance from the
    ##membrane is computed. A top-hat filter is applied to find the maximum
    ##within a given radius and a distance threshold is applied to weed out
    ##seeds near the membrane.
    ##'''
    ##DistanceTransform=2
    
    ##'''Use connected components instead of finding seeds
    
    ##The probabilities are thresholded and connected components is run
    ##on pixels lower than the threshold.
    ##'''
    ##ConnectedComponents=3


##class Dimensionality(enum.Enum):
    ##'''Determines whether to do something in 2D or 3D'''
    
    ##'''Process a 3d volume as 2d planes'''
    ##D2=2
    
    ##'''Process a 3d volume as a whole'''
    ##D3=3

##class Shape(enum.Enum):
    ##'''Determines the shape of the erosion structuring element'''
    
    ##'''Ellipsoid - truly based on distance, but not separable and slower'''
    ##Ellipsoid=1
    
    ##'''Cube - includes points outside of ellipsoid but separable and faster'''
    ##Cube=2
    
##class FindSeedsTaskMixin:
    
    ##volume = VolumeParameter(
        ##description="The volume being segmented")
    ##prob_location = DatasetLocationParameter(
        ##description="The location of the membrane probabilities")
    ##seeds_location = DatasetLocationParameter(
        ##description="The location of the volume with seed labels")
    
    ##def input(self):
        ##yield TargetFactory().get_volume_target(
            ##location=self.prob_location, volume=self.volume)
    
    ##def output(self):
        ##return TargetFactory().get_volume_target(
            ##location=self.seeds_location, volume=self.volume)

    ##def estimate_memory_usage(self):
        ##'''Return an estimate of bytes of memory required by this task'''
        ##v1 = np.prod([1416, 1888, 70])
        ##m1 = 2877614 * 1000
        ##v2 = np.prod([1416, 1888, 42])
        ##m2 = 1693022 * 1000
        ###
        ### Model is Ax + B where x is volume in voxels
        ###
        ##B = (v1 * m2 - v2 * m1) / (v1 - v2)
        ##A = (float(m1) - B) / v1
        ##v = np.prod([self.volume.width, self.volume.height, self.volume.depth])
        ##return int(A * v + B)


class FindSeedsRunMixin:
    
    def __init__(self, pdimensionality, pmethod):
        self.dimensionality = pdimensionality #"Whether to find seeds in each 2D plane or in the volume as a whole")
        self.method = pmethod #"The algorithm used to find seeds: Smoothing, DistanceTransform"
        self.sigma_xy = 1.5 #"The sigma of the smoothing Gaussian in the x & y directions",
        self.sigma_z = 0.4  #"The sigma of the smoothing Gaussian in the z direction",
        self.threshold = 10 #"The intensity threshold cutoff for the seeds",
        self.minimum_distance_xy = 10.0 #"The minimum distance allowed between seeds"
        self.minimum_distance_z = 2.0 #"The minimum distance allowed between seed in the z dir"
        
        self.structuring_element = 'Cube' 
        #"The shape of the structuring element. Ellipsoid is slower, but honors the distances.
        # Cube is faster, but excludes due to extrema at the corners of the cube")
        
        
        self.distance_threshold = 20 #"The distance threshold cutoff for the seeds in nm"
        #
        # Parameters for block management of the distance threshold calculation
        #
        self.xy_nm = 4.0  #"Size of a voxel in the X and Y direction"
        self.z_nm = 30.0 #"Size of a voxel in the Z direction"
        self.dt_xy_overlap = 40 #"Overlap between distance transform blocks in the x and y directions"
        self.dt_z_overlap = 5 #"Overlap between distance transform blocks in the z direction"
        self.dt_xy_block_size = 512 #"Block size in the x and y directions for the distance transform."
        self.dt_z_block_size = 40 #"Block size in the z direction for the distance transform"
        self.dt_n_cpus = 4 #"Number of CPUs to use when computing the distance transform"
    
    def set_threshold(self, pthd):
        self.threshold = pthd
    
    def make_strel(self):
        '''make the structuring element for the minimum distance'''
        if self.structuring_element == 'Cube':
            return np.ones([int(np.floor(_) * 2 + 1) for _ in
                            self.minimum_distance_z,
                            self.minimum_distance_xy,
                            self.minimum_distance_xy], bool)
        
        ixy = int(np.floor(self.minimum_distance_xy))
        iz = int(np.floor(self.minimum_distance_z))
        z, y, x = np.mgrid[-iz:iz+1, -ixy:ixy+1, -ixy:ixy+1].astype(np.float32)
        strel = ((z / self.minimum_distance_z) ** 2 +
                 (y / self.minimum_distance_xy) ** 2 +
                 (x / self.minimum_distance_xy) ** 2) <= 1
        return strel
    
    def find_using_2d_smoothing(self, probs):
        '''Find seeds in each plane, smoothing, then thresholding
        
        :param probs: the probability volume
        '''
        
        if (self.dimensionality!= 2):
            print "Wrong dimensionality"
            return
        if (self.method!="Smoothing"):
            print "Wrong method"
            return

        
        offset=0
        seeds = []
        for plane in probs.astype(np.float32):
            smoothed = gaussian_filter(plane.astype(np.float32), self.sigma_xy)
            size = self.minimum_distance_xy
            eroded = grey_erosion(smoothed, size)
            thresholded = (smoothed < self.threshold) & (smoothed == eroded)
            labels, count = label(thresholded)
            labels[labels != 0] += offset
            offset += count
            seeds.append(labels)
        return np.array(seeds)
    
    def find_using_3d_smoothing(self, probs):
        '''Find seeds after smoothing and thresholding

        :param probs: the probability volume
        '''

        if (self.dimensionality!= 3):
            print "Wrong dimensionality"
            return
        if (self.method!="Smoothing"):
            print "Wrong method"
            return
        
        sigma = (self.sigma_z, self.sigma_xy, self.sigma_xy)
        smoothed = gaussian_filter(probs.astype(np.float32), sigma)
        eroded = grey_erosion(smoothed, footprint=self.make_strel())
        thresholded = (smoothed < self.threshold) & (smoothed == eroded)
        labels, count = label(thresholded)
        #rh_logger.logger.report_event("Found %d seeds" % count)
        return labels
    
    def find_using_2d_distance(self, probs):
        '''Find seeds in each plane by distance transform

        :param probs: the probability volume
        '''

        if (self.dimensionality!= 2):
            print "Wrong dimensionality"
            return
        if (self.method!="DistanceTransform"):
            print "Wrong method"
            return

        offset=0
        seeds = []
        for plane in probs.astype(np.float32):
            thresholded = plane < self.threshold
            distance = distance_transform_edt(thresholded)
            dilated = grey_dilation(distance, size=self.minimum_distance_xy)
            mask = (distance == dilated) & (distance >= self.distance_threshold)
            labels, count = label(mask)
            labels[labels != 0] += offset
            offset += count
            seeds.append(labels)
        return np.array(seeds)
    
    def find_using_3d_distance(self, probs):
        
        if (self.dimensionality!= 3):
            print "Wrong dimensionality"
            return
        if (self.method!="DistanceTransform"):
            print "Wrong method"
            return
        
        distance = []
        thresholded = probs < self.threshold
        distance = parallel_distance_transform(
            thresholded, self.xy_nm, self.z_nm, 
            self.dt_xy_overlap, self.dt_z_overlap, 
            self.dt_xy_block_size, self.dt_z_block_size, self.dt_n_cpus)
        dilated = grey_dilation(distance, footprint=self.make_strel())
        mask = (distance == dilated) & (distance >= self.distance_threshold)
        labels, count = label(mask)
        rh_logger.logger.report_event("Found %d seeds" % count)
        return labels
        
    ##def ariadne_run(self):
        ##prob_target = self.input().next()
        ##probs = prob_target.imread()
        ##if self.method == SeedsMethodEnum.Smoothing:
            ##if self.dimensionality == Dimensionality.D2:
                ##seeds = self.find_using_2d_smoothing(probs)
            ##else:
                ##seeds = self.find_using_3d_smoothing(probs)
        ##else:
            ##if self.dimensionality == Dimensionality.D2:
                ##seeds = self.find_using_2d_distance(probs)
            ##else:
                ##seeds = self.find_using_3d_distance(probs)
        ##seeds = seeds.astype(np.uint32)
        ##self.output().imwrite(seeds)


##class FindSeedsTask(FindSeedsTaskMixin,
                    ##FindSeedsRunMixin,
                    ##RequiresMixin,
                    ##RunMixin,
                    ##SingleThreadedMixin,
                    ##luigi.Task):
    ##'''Find seeds for a watershed.
    
    ##This task takes a probability map as input and produces a "segmentation"
    ##where the seeds for a watershed are labeled, using a different index
    ##per seed.
    
    ##You can choose between different algorithms using the "method" parameter
    ##and you can either find seeds in each 2D plane or in the whole 3d
    ##volume.
    ##'''
    ##task_namespace = "ariadne_microns_pipeline"
