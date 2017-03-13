'''Script to compute the NRI

This is an implementation of the paper: https://arxiv.org/pdf/1702.02684.pdf

Neural Reconstruction Integrity: A metric for assessing the
connectivity of reconstructed neural networks, Elizabeth Reilly

To see arguments:

python nri.py --help 

The NRI is computed from

* segmentation-file
* synapse-segmentation-file
* pre-synaptic-map - the classifier map specifying the probability that a pixel
                     is in the presynaptic portion of a synapse
* ground-truth-file - the ground-truth segmentation
* ground-truth-synapse-file - the ground-truth synapses. 0 is background, 1 is
                              presynaptic, 2 is postsynaptic.

You can save volumes from the pipeline to .h5 using the command, 
"microns-volume" (part of the microns-ariadne/pipeline_engine package).
'''
import argparse
import h5py
import hungarian
import numpy as np
from scipy.ndimage import label, grey_dilation, binary_dilation, grey_erosion
from scipy.sparse import coo_matrix

#
# From the paper, the maximum distance in nm between matching synapse centroids
#
MAX_SYNAPSE_DISTANCE = 300
#
# The size of a voxel
#
xy_nm = 4.0
z_nm = 30.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-file",
                        help="The .h5 file containing the segmentation of the volume")
    parser.add_argument("--synapse-segmentation-file",
                        help="The .h5 file containing the segmentation of the synapses")
    parser.add_argument("--pre-synaptic-map-file",
                        help="The .h5 file containing the probability map "
                        "classifying a voxel as being part of the presynaptic "
                        "side of the synapse")
    parser.add_argument("--ground-truth-file",
                        help="The .h5 file containing the ground-truth segmentation")
    parser.add_argument("--ground-truth-synapse-file",
                        help="The .h5 file containing the ground-truth synapse annotations")
    return parser.parse_args()

def c2(x):
    '''Compute N choose 2 for x'''
    return x * (x-1)

def get_gt_connections(gt_file, gt_synapse_file):
    '''Match gt synapses against gt neurons
    
    Output is a dictionary of the same form as the synapse connections file
    '''
    with h5py.File(gt_synapse_file, "r") as fd:
        synapse_pre = fd[fd.keys()[0]][:] == 1
        synapse_post = fd[fd.keys()[0]][:] == 2
    #
    # Find correspondences between pre-synaptic and post-synaptic label partners
    #
    labels_pre, pre_count = label(synapse_pre)
    labels_post, post_count = label(synapse_post)
    #
    # Dilate to create some overlap
    #
    dlabels_pre = grey_dilation(labels_pre, size=3)
    dlabels_post = grey_dilation(labels_post, size=3)
    #
    # Find where the overlap is
    #
    overlap_z, overlap_y, overlap_x = np.where(dlabels_pre * dlabels_post > 0)
    #
    # Count correspondences between labels
    #
    matrix = coo_matrix((np.ones(len(overlap_z)), 
                         (dlabels_pre[overlap_z, overlap_y, overlap_x],
                          dlabels_post[overlap_z, overlap_y, overlap_x])))
    del dlabels_pre
    del dlabels_post
    matrix.sum_duplicates()
    matrix = matrix.toarray()
    #
    # Pull out the best "post" per pre. Assume that they *are* matched
    #
    pre_id = np.arange(1, pre_count+1)
    post_id = np.argmax(matrix[1:, 1:], 1) + 1
    #
    # OK, so don't assume.
    #
    mask = matrix[pre_id, post_id] > 100
    pre_id, post_id = pre_id[mask], post_id[mask]
    #
    # Now match neurons against synapses
    #
    with h5py.File(gt_file, "r") as fd:
        gt = fd[fd.keys()[0]][:]
    #
    # presynaptic neuron first
    #
    overlap_z, overlap_y, overlap_x = np.where(gt * labels_pre > 0)
    matrix = coo_matrix((np.ones(len(overlap_z)), 
                         (labels_pre[overlap_z, overlap_y, overlap_x],
                          gt[overlap_z, overlap_y, overlap_x])))
    matrix.sum_duplicates()
    matrix = matrix.toarray()
    npre_id = np.argmax(matrix[pre_id, 1:], 1) + 1
    #
    # postsynaptic neuron next
    #
    overlap_z, overlap_y, overlap_x = np.where(gt * labels_post > 0)
    matrix = coo_matrix((np.ones(len(overlap_z)), 
                         (labels_post[overlap_z, overlap_y, overlap_x],
                          gt[overlap_z, overlap_y, overlap_x])))
    matrix.sum_duplicates()
    matrix = matrix.toarray()
    npost_id = np.argmax(matrix[post_id, 1:], 1) + 1
    #
    # and finally, the x, y and z
    #
    z, y, x = np.where(labels_pre > 0)
    areas = np.bincount(labels_pre[z, y, x])[1:]
    xc = np.bincount(labels_pre[z, y, x], x)[1:] / areas
    yc = np.bincount(labels_pre[z, y, x], y)[1:] / areas
    zc = np.bincount(labels_pre[z, y, x], z)[1:] / areas
    return dict(neuron_1=npre_id,
                neuron_2=npost_id,
                synapse_center=dict(x=xc[mask], y=yc[mask], z=zc[mask]))

def get_connections(seg_file, syn_seg_file, pre_synaptic_probs):
    '''Get the detected synaptic connections
    
    :param seg_file: the .h5 segmentation
    :param syn_seg_file: the .h5 segmentation of the synapses
    :param pre_synaptic_probs: the .h5 probability maps classifying
    voxels as pre-synaptic
    '''
    #
    # The strategy:
    #
    # * sum the probability map within the synapse regions to get
    #   the average strength of the signal within each neuron
    # * find only the border pixels of the segmentations
    # * overlay with synapses to get only border pixels within synapses
    # * use np.bincount to compute average x, y and z
    #
    with h5py.File(seg_file, "r") as fd:
        seg_volume = fd[fd.keys()[0]][:]
    with h5py.File(syn_seg_file, "r") as fd:
        synseg_volume = fd[fd.keys()[0]][:]
    ############################################
    #
    # Find the neuron pairs.
    #
    ############################################
    z, y, x = np.where(synseg_volume > 0)
    seg, synseg = seg_volume[z, y, x], synseg_volume[z, y, x]
    matrix = coo_matrix((np.ones(len(z)), (synseg, seg)))
    matrix.sum_duplicates()
    synapse_labels, neuron_labels = matrix.nonzero()
    counts = matrix.tocsr()[synapse_labels, neuron_labels].getA1()
    #
    # Order by synapse label and -count to get the neurons with
    # the highest count first
    #
    order = np.lexsort((-counts, synapse_labels))
    counts, neuron_labels, synapse_labels = \
        [_[order] for _ in counts, neuron_labels, synapse_labels]
    first = np.hstack(
        [[True], synapse_labels[:-1] != synapse_labels[1:], [True]])
    idx = np.where(first)[0]
    per_synapse_counts = idx[1:] - idx[:-1]
    #
    # Get rid of counts < 2
    #
    mask = per_synapse_counts >= 2
    idx = idx[:-1][mask]
    #
    # pick out the first and second most overlapping neurons and
    # their synapse.
    #
    neuron_1 = neuron_labels[idx]
    synapses = synapse_labels[idx]
    neuron_2 = neuron_labels[idx+1]
    ###################################
    # 
    # Determine polarity
    #
    ###################################
    with h5py.File(pre_synaptic_probs, "r") as fd:
        probs = fd[fd.keys()[0]][:][z, y, x]
    #
    # Start by making a matrix to transform the map.
    #
    matrix = coo_matrix(
        (np.arange(len(idx)*2) + 1,
         (np.hstack((neuron_1, neuron_2)),
          np.hstack((synapses, synapses)))),
        shape=(np.max(seg)+1, np.max(synseg) + 1)).tocsr()
    #
    # Convert the neuron / synapse map to the mapping labels
    #
    mapping_labeling = matrix[seg, synseg].A1
    #
    # Score each synapse / label overlap on both the transmitter
    # and receptor probabilities
    #
    areas = np.bincount(mapping_labeling)
    transmitter_score = np.bincount(
            mapping_labeling, probs, minlength=len(areas)) / areas
    del probs
    score_1 = transmitter_score[1:len(idx)+1]
    score_2 = transmitter_score[len(idx)+1:]
    #
    # Flip the scores and neuron assignments if score_2 > score_1
    #
    flippers = score_2 > score_1
    score_1[flippers], score_2[flippers] = \
        score_2[flippers], score_1[flippers]
    neuron_1[flippers], neuron_2[flippers] = \
        neuron_2[flippers], neuron_1[flippers]
    ##########################################################
    #
    # Compute synapse centers
    #
    ##########################################################
    edge_z, edge_y, edge_x = np.where(
        (grey_dilation(seg, size=3) != grey_erosion(seg_volume, size=3)) &\
        (synseg_volume != 0))
    areas = np.bincount(synseg_volume[edge_z, edge_y, edge_x])
    xc, yc, zc = [np.bincount(synseg_volume[edge_z, edge_y, edge_x], _)
                  for _ in edge_x, edge_y, edge_z]
    result = dict(neuron_1=neuron_1,
                  neuron_2=neuron_2,
                  synapse_center=dict(x=xc[synapses]/areas[synapses],
                                      y=yc[synapses]/areas[synapses],
                                      z=zc[synapses]/areas[synapses]))
    return result
    
def match_synapses(connections, gt_connections):
    '''Match ground-truth synapses against detected synapses
    
    Returns the count table calculation from table 2 of the paper.
    '''
    #
    # Make a matrix of distances and augment the matrix with big numbers
    # so that is square. Things that match the augmented side get the booby
    # prize of being inserts or deletes.
    #
    # The NDI paper says that the max distance is 300 nm
    #
    x = np.array(connections["synapse_center"]["x"])
    y = np.array(connections["synapse_center"]["y"])
    z = np.array(connections["synapse_center"]["z"])
    n1 = np.array(connections["neuron_1"])
    n2 = np.array(connections["neuron_2"])
    
    gtx = gt_connections["synapse_center"]["x"]
    gty = gt_connections["synapse_center"]["y"]
    gtz = gt_connections["synapse_center"]["z"]
    gtn1 = gt_connections["neuron_1"]
    gtn2 = gt_connections["neuron_2"]
    #
    # First, we set the matrix to have the augmented value
    #
    side = len(x) + len(gtx)
    matrix = np.ones((side, side), int) * MAX_SYNAPSE_DISTANCE
    #
    # Then we place the distances within
    #
    matrix[:len(x), :len(gtx)] = np.sqrt(
        ((x[:, np.newaxis] - gtx[np.newaxis, :]) * xy_nm) ** 2 +
        ((y[:, np.newaxis] - gty[np.newaxis, :]) * xy_nm) ** 2 +
        ((z[:, np.newaxis] - gtz[np.newaxis, :]) * z_nm) **2).astype(int)
    #
    # Run the hungarian
    #
    detected_id, gt_id = hungarian.lap(matrix)
    #
    # Get rid of the augmented portion of the matches
    #
    detected_id = detected_id[:len(x)]
    gt_id = gt_id[:len(gtx)]
    #
    # If detected_id is in the augmented range, it's an insertion and goes
    # into row 0
    #
    insertion_idxs = np.where(detected_id >= len(gtx))[0]
    insertion_ids = np.hstack((n1[insertion_idxs], n2[insertion_idxs]))
    #
    # Likewise with gt_id and deletions
    #
    deletion_idxs = np.where(gt_id >= len(x))[0]
    deletion_ids = np.hstack((gtn1[deletion_idxs], gtn2[deletion_idxs]))
    #
    # The matches
    #
    idx = np.where(detected_id < len(x))[0]
    n1_ids = n1[idx]
    n2_ids = n2[idx]
    gtn1_ids = gtn1[detected_id[idx]]
    gtn2_ids = gtn2[detected_id[idx]]
    #
    # And put it all together into the matrix to return
    #
    d = np.hstack((insertion_ids, np.zeros(len(deletion_ids), int), 
                   n1_ids, n2_ids))
    gt = np.hstack((np.zeros(len(insertion_ids), int), deletion_ids, 
                    gtn1_ids, gtn2_ids))
    matrix = coo_matrix((np.ones(len(d)), (d, gt)))
    matrix.sum_duplicates()
    return matrix.toarray()
    
def main():
    args = parse_args()
    #
    # Step 1 - get the ground truth connections
    #
    gt_connections = get_gt_connections(args.ground_truth_file,
                                        args.ground_truth_synapse_file)
    #
    # Step 2 - find the detected connections
    #
    connections = get_connections(args.segmentation_file,
                                  args.synapse_segmentation_file,
                                  args.pre_synaptic_map_file)
    #
    # Step 3 - find synapse correspondences
    #
    matrix = match_synapses(connections, gt_connections)
    #
    # Step 4 - compute the NRI from the matrix and # of inserts and deletions
    #
    # Eqn # 6
    #
    # TP = sum over ij of matrix[i,j] choose 2 =
    # choose 2 = n * (n - 1)
    #
    tp = np.sum(c2(matrix[1:, 1:]))
    #
    # Eqn # 7
    #
    fn = sum(c2(matrix[:, 0]))
    for i in range(1, matrix.shape[0]):
        row = matrix[i, :]
        nz = row[row != 0]
        fn += np.sum(nz[:, np.newaxis] * nz[np.newaxis, :] * 
                     (1 - np.eye(len(nz)))) / 2
    fp = sum(c2(matrix[0, :]))
    for j in range(1, matrix.shape[1]):
        col = matrix[:, j]
        nz = col[col != 0]
        fp += np.sum(nz[:, np.newaxis] * nz[np.newaxis, :] * 
                     (1 - np.eye(len(nz)))) / 2
    #
    # Eqn 1
    #
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    nri = 2 * (precision * recall) / (precision + recall)
    
    print "True positive:  %d" % tp
    print "False positive: %d" % fp
    print "False negative: %d" % fn
    print "Precision:      %.1f %%" % (precision * 100)
    print "Recall:         %.1f %%" % (recall * 100)
    print "NRI:            %.3f" % nri
    
if __name__ == "__main__":
    main()