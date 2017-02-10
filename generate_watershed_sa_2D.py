import h5py
import numpy as np
import find_seeds_sa as FS
#from skimage import morphology as skmorph
import vigra
import pdb


def relabel_from_one(a):
    labels = np.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return a, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0)+1)
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map

def remove_small_connected_components(a, min_size=64, in_place=False):
    original_dtype = a.dtype
    if a.dtype == bool:
        a = label(a)[0]
    elif not in_place:
        a = a.copy()
    if min_size == 0: # shortcut for efficiency
        return a
    component_sizes = np.bincount(a.ravel())
    too_small = component_sizes < min_size
    too_small_locations = too_small[a]
    a[too_small_locations] = 0
    return a.astype(original_dtype)


if __name__ == "__main__" :
    
    
    
    fid = h5py.File('/home/paragt/work/pipeline_codes/train_vol/membrane.h5')
    pp=np.array(fid['stack'])
    fid.close()
    
    
    # creating a dummy channel because I am not good at coding :)
    pp_save = np.zeros(np.concatenate((pp.shape,[2]),axis=0)).astype(np.float32)
    pp_save[...,0] = pp*1.0/np.amax(pp)
    pp_save[...,1] = 1 - pp_save[...,0]
    
    
    fidw = h5py.File('/home/paragt/work/pipeline_codes/train_vol/pixel_pred1.h5','w')
    fidw.create_dataset('stack',data=pp_save.astype(np.float32))
    fidw.close()    
    
    compute_seeds=FS.FindSeedsRunMixin(2,'Smoothing')
    seeds_2D=compute_seeds.find_using_2d_smoothing(pp)
    print "Computed  "+str(len(np.unique(seeds_2D)))+" markers"
    pdb.set_trace()
    
    ws3d = np.zeros(pp.shape).astype(np.uint32)
    min_segid=1
    for plane in range(0,pp.shape[0]):
        
        
        
        ws,dummy=vigra.analysis.watersheds(pp[plane,:,:].astype(np.uint8),seeds=seeds_2D[plane,:,:].astype(np.uint32))
    
        print "Computed watershed with "+str(len(np.unique(ws)))+" regions"
    
        ##seeds = label(pp[plane,:,:]<0.05)[0] # 0.3 for Dan cirecan, 0.01 for ours, 0.1 for jan funke
        ##if seed_cc_threshold > 0:
            ##seeds = morpho.remove_small_connected_components(seeds, seed_cc_threshold)
        ##ws = skmorph.watershed(pp[plane,:,:], seeds)
        ##print "Imported first stack with "+ str(unique(ws).size) + " regions"
        
        ws_int, dummy1, dummy2 = relabel_from_one(ws)
        max_id = np.amax(ws_int)    
        ws_int = ws_int + min_segid
        ws_int1 =  ws_int.astype(np.uint32)
        if len(ws_int1.shape)<3:
            ws_int1 = np.expand_dims(ws_int1, axis=0)
            
        min_segid = min_segid + max_id
        ws3d[plane,:,:] = ws_int1
     
    
    print "Computed watershed with "+str(len(np.unique(ws3d)))+" regions"
    fidw = h5py.File('/home/paragt/work/pipeline_codes/train_vol/watershed_vigra_2D.h5','w')
    fidw.create_dataset('stack',data=ws3d.astype(np.uint32))
    fidw.close()