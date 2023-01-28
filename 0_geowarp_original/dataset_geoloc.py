
import os
import numpy as np
from glob import glob
from os.path import join
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors

import datasets_util

# DATASET PER IL TRAINING DATASET T COMPOSTA DA GALLERY E DALLE QUERY

class GeolocDataset(data.Dataset):     # questo è nostr dataset T contennte la gallery e le query, in questo caso pitts30k
    def __init__(self, datasets_folder="datasets", dataset_name="pitts30k", split="train", positive_dist_threshold=25):
        """
        
        Parameters
        ----------
        datasets_folder : str, path of the folder with the datasets.
        dataset_name : str, name of the folder with the dataset within datasets_folder.
        split : str, split to use among train, val or test.
        positive_dist_threshold : int, the threshold for positives (in meters).
        
        The images should be located at these two locations:
            {datasets_folder}/{dataset_name}/images/{split}/gallery/*.jpg
            {datasets_folder}/{dataset_name}/images/{split}/queries/*.jpg
        
        """
        
        super().__init__()
        self.dataset_name = dataset_name                                                          # nome del dataset
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)                # cartella del dataset
        if not os.path.exists(self.dataset_folder):                                               # controllo se la cartella esiste
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        #### Read paths and UTM coordinates for all images.                                       # il dataset è diviso in due parti, gallery e queries
        gallery_folder = join(self.dataset_folder, "gallery")                                     # abbiamo la gallery
        queries_folder = join(self.dataset_folder, "queries")                                     # abbiamo le query
        if not os.path.exists(gallery_folder):                                                    # controllo sulle due cartelle
            raise FileNotFoundError(f"Folder {gallery_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.gallery_paths = sorted(glob(join(gallery_folder, "**", "*.jpg"), recursive=True))    # join: unisce le stringhe, glob: ritorna tutti i path che soddisfano il pattern, sorted: ordina i path
        self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))    # stessa cosa per le query
        
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.gallery_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.gallery_paths]).astype(np.float)    # prende le coordinate delle gallery
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)    # prende le coordinate delle query
        
        # Find soft_positives_per_query, which are within positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)                                                     # usa Knn per ragguppare i positivi delle query(tutte le query in un certo range)
        knn.fit(self.gallery_utms)                                                            # fit il modello sugli utms
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,               # prende tutti i positive delle query
                                                             radius=positive_dist_threshold,
                                                             return_distance=False)
        self.images_paths = list(self.gallery_paths) + list(self.queries_paths)               # unico path per le immagini
        
        self.gallery_num = len(self.gallery_paths)                                            # conta le immagini (path diversi)
        self.queries_num = len(self.queries_paths)                                            # conta le query (path diversi)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        img = datasets_util.open_image_and_apply_transform(image_path)
        return img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #gallery: {self.gallery_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):                     # ritorna i positives per query
        return self.soft_positives_per_query
