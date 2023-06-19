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

    regions_number = original_image_size//attn_scores.shape[1]
    x_loc = x_coords*regions_number+regions_number//2
    y_loc = y_coords*regions_number+regions_number//2

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



# Il KDTree viene utilizzato per accelerare la ricerca dei descrittori più vicini tra immagini diverse, 
# mentre RANSAC viene utilizzato per eliminare i descrittori che non appartengono a un modello coerente, 
# come i descrittori errati o outliers. Il modello coerente identificato da RANSAC viene quindi utilizzato 
# per effettuare la corrispondenza dei descrittori e calcolare la trasformazione tra le immagini. 
# Questo processo aiuta a migliorare la precisione e la stabilità delle corrispondenze tra le immagini.

def match_features(query_locations,
                  query_descriptors,
                  image_locations,
                  image_descriptors,
                  descriptor_matching_threshold=0.6,
                  ransac_residual_threshold=15.0,
                  query_im_array=None,
                  index_im_array=None,
                  ransac_seed=None,
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



# La stima di una trasformazione affine fra due set di punti in 2D può avere diverse applicazioni nel contesto del 
# processamento delle immagini e della visione artificiale.

# Uno degli utilizzi più comuni è quello della registrazione delle immagini, che consiste nell'allineare due o più 
# immagini dello stesso soggetto prese da angolazioni o tempi diversi. Nella registrazione delle immagini, la trasformazione 
# affine permette di correggere le variazioni di scala, rotazione, traslazione e deformazione (shear) tra le immagini.

# Nel contesto del tuo codice, l'algoritmo sembra essere utilizzato per trovare corrispondenze tra feature locali 
# (descrittori) in due immagini diverse. Le corrispondenze vengono poi utilizzate per stimare una trasformazione 
# affine che mappa le posizioni delle feature nell'immagine di query alle posizioni corrispondenti nell'immagine del database.

# Questo può essere utile in una varietà di contesti, come il riconoscimento di oggetti o luoghi, la creazione di 
# panorami da più immagini, o la realtà aumentata. Per esempio, se si sta cercando di riconoscere un luogo da una foto, 
# si può cercare di mappare le feature della foto a quelle di immagini note dello stesso luogo nel database. Se si può trovare 
# una buona trasformazione affine che mappa le feature della foto a quelle delle immagini del database, allora è probabile che 
# la foto rappresenti lo stesso luogo.

# Inoltre, la trasformazione affine stessa può fornire informazioni utili. Per esempio, la scala e la rotazione della 
# trasformazione possono dare un'idea della differenza di orientamento e distanza tra la camera e il soggetto nelle due immagini.


#Se stai cercando di allineare le caratteristiche tra due immagini specifiche, allora sì, avrai bisogno di calcolare una 
# trasformazione affine per ogni coppia di immagini. Questo perché ogni coppia di immagini avrà la propria relazione unica 
# di traslazione, rotazione, scala e deformazione, determinata dalla posizione e orientamento relativo della camera quando 
# ciascuna immagine è stata scattata.

# Non è possibile calcolare una singola trasformazione affine che sarà valida per tutte le possibili immagini, a meno che non ci 
# sia qualche condizione molto specifica che renda ciò possibile. Per esempio, se stai lavorando con un set di immagini che sono 
# state tutte prese dalla stessa posizione e orientamento, allora potresti essere in grado di utilizzare la stessa trasformazione 
# affine per tutte le immagini. Ma in generale, questo non sarà il caso.

# Inoltre, anche se fosse possibile calcolare una singola trasformazione affine per tutte le immagini, ciò potrebbe non essere 
# desiderabile. Una delle principali vantaggi del RANSAC e della stima di una trasformazione affine è che sono robusti agli outlier. 
# Se si calcola una singola trasformazione affine per tutte le immagini, si rischia di ottenere una trasformazione che è distorta 
# da outlier nelle immagini.

# In conclusione, nel tuo caso, calcolare una trasformazione affine per ogni coppia di immagini sembra essere l'approccio più corretto.