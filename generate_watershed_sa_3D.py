import h5py
import numpy as np
import find_seeds_sa as FS
from scipy.ndimage import label
#from skimage import morphology as skmorph
import vigra
import pdb

if __name__ == "__main__" :
    
    
    
    fid = h5py.File('/home/paragt/work/pipeline_codes/train_vol/membrane.h5')
    pp=np.array(fid['stack'])
    fid.close()
    
    
    pp_save = np.zeros(np.concatenate((pp.shape,[2]),axis=0)).astype(np.float32)
    pp_save[...,0]=pp
    pp_save[...,1]=1-pp
    
    fidw = h5py.File('/home/paragt/work/pipeline_codes/train_vol/pixel_pred.h5','w')
    fidw.create_dataset('stack',data=pp_save.astype(np.float32))
    fidw.close()    
    
    compute_seeds=FS.FindSeedsRunMixin(3,'Smoothing')

    seeds_3D=compute_seeds.find_using_3d_smoothing(pp)
    print "Computed  "+str(len(np.unique(seeds_3D)))+" markers"
    
    ##bpp = (pp<0.1)
    ##seeds_3D,nmarkers = label(bpp)
    ##print "Computed  "+str(nmarkers)+" markers"
    
    ws,max_id=vigra.analysis.watersheds(pp.astype(np.uint8),seeds=seeds_3D.astype(np.uint32))
    
    print "Computed watershed with "+str(max_id)+" regions"
    
    fidw = h5py.File('/home/paragt/work/pipeline_codes/train_vol/watershed_vigra.h5','w')
    fidw.create_dataset('stack',data=ws.astype(np.uint32))
    fidw.close()