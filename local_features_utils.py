import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from skimage import feature
from skimage import transform
from skimage.measure import ransac
import torch.nn.functional as F
import warnings

def retrieve_locations_and_descriptors(attn_scores, red_feature_map, original_image_size=224, k=85):
    attn_scores = np.transpose(attn_scores, (1, 2, 0))
    red_feature_map = np.transpose(red_feature_map, (1, 2, 0))
    flat_indices = np.argpartition(attn_scores.flatten(), -k)[-k:]
    sorted_indices = flat_indices[np.argsort(attn_scores.flatten()[flat_indices])][::-1]  
    k_coords = np.unravel_index(sorted_indices, attn_scores.shape)  
    x_coords, y_coords = k_coords[:-1]

    local_descriptors_ij = list(zip(x_coords, y_coords)) 

    region_size = original_image_size//attn_scores.shape[1]
    x_loc = x_coords*region_size+region_size//2
    y_loc = y_coords*region_size+region_size//2

    local_descriptors_locations = np.array(list(zip(x_loc, y_loc)))

    filtered_local_descriptors = []
    for i, j in local_descriptors_ij:
      filtered_local_descriptors.append(red_feature_map[i, j, :])
    local_descriptors = np.array(filtered_local_descriptors)
    
    local_descriptors = torch.from_numpy(local_descriptors)
    local_descriptors = F.normalize(local_descriptors, p=2, dim=1).numpy()
    return local_descriptors_locations, local_descriptors


def plot_inlier_lines(query_im_array, index_im_array, query_locations, index_locations, inliers):
    inlier_idxs = np.nonzero(inliers)[0]
    
    _, ax = plt.subplots()
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    feature.plot_matches(ax, query_im_array, 
                         index_im_array, 
                         query_locations, 
                         index_locations,
                         np.column_stack((inlier_idxs, inlier_idxs)), 
                         only_matches=True,)

    plt.show()


def match_features(query_locations,
                  query_descriptors,
                  image_locations,
                  image_descriptors,
                  descriptor_matching_threshold=0.65,
                  ransac_residual_threshold=15.0,
                  query_im_array=None,
                  index_im_array=None,
                  ransac_seed=4,
                  use_ratio_test=False,
                  RANSAC=True):


    num_features_query = query_locations.shape[0]    
    num_features_image = image_locations.shape[0]
    if not num_features_query or not num_features_image:
        print(f"database images or query image don't have consistent dimension")

    local_feature_dim = query_descriptors.shape[1]       
    if image_descriptors.shape[1] != local_feature_dim:
        print(f"Local feature dimensionality is not consistent for query and database images.")

    index_image_tree = spatial.cKDTree(image_descriptors)
    if use_ratio_test:
        distances, indices = index_image_tree.query(query_descriptors, k=2, workers=-1)

        query_locations_to_use = np.array([
            query_locations[i,]
            for i in range(num_features_query)
            if distances[i][0] < descriptor_matching_threshold * distances[i][1]
        ])
        image_locations_to_use = np.array([
            image_locations[indices[i][0],]
            for i in range(num_features_query)
            if distances[i][0] < descriptor_matching_threshold * distances[i][1]
        ])
    else:
        _, indices = index_image_tree.query(query_descriptors, distance_upper_bound=descriptor_matching_threshold, workers=-1)
        query_locations_to_use = np.array([query_locations[i,] for i in range(num_features_query) if indices[i] != num_features_image])
        
        image_locations_to_use = np.array([image_locations[indices[i],] for i in range(num_features_query) 
                                    if indices[i] != num_features_image])
    
    if RANSAC:
        _NUM_RANSAC_TRIALS = 100
        _MIN_RANSAC_SAMPLES = 3

        if query_locations_to_use.shape[0] <= _MIN_RANSAC_SAMPLES:
            return 0
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'No inliers found. Model not fitted', UserWarning)
            _, inliers = ransac(
                (image_locations_to_use, query_locations_to_use),
                transform.AffineTransform,
                min_samples=_MIN_RANSAC_SAMPLES,
                residual_threshold=ransac_residual_threshold,
                max_trials=_NUM_RANSAC_TRIALS,
                random_state=ransac_seed)

        if inliers is None:
            inliers = []

        elif query_im_array is not None and index_im_array is not None:
            plot_inlier_lines(query_im_array, index_im_array, query_locations_to_use, image_locations_to_use, inliers) 

        return sum(inliers)
    else:
        return image_locations_to_use.shape[0]


