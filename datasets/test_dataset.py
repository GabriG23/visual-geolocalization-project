
import os
import numpy as np
from glob import glob
from os.path import join  
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database", queries_folder="queries", positive_dist_threshold=25):         # positive_dist_threshold viene passato come argomento da tastiera
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)    # concatena il path del secondo elemento (che è solo un nome) a quello del primo (che è più lungo)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)                    # resituisce la parte finale del path (cartella o file)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")     # errori vari se le cartelle non esistono
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # stessa mean e std del train
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))   # prende i path in ordine alfabetico che matchano
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)  # prende  utmeast e utmnorth
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)           # da sklearn.neighbors. Restituisce un oggetto in grado di implementare neighbor searches. n_jobs=-1 significa che userà
                                                    # tutti i processori.
        knn.fit(self.database_utms)                 # allena il NearestNeighbors con le immagini del database
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,                  # trova i vicini all'interno di un dato raggio (25 mt di defualt) con centro un punto (la query) e ne ritorna 
                                                        radius=positive_dist_threshold,     # gli indici (solo gli indici e non le distanze perché return_distance=False)
                                                        return_distance=False)

        # Da quel che ho capito, il NearestNeighbors viene allenato sul dataset in cui ogni sample è un vettore (utmeast, utmnorth). Dopodiché vengono inserite le query come samples di test
        # e per ogni dato di test (quindi ogni query), all'interno del raggio dato vengono restituiti gli indici dei sample del dataset vicini (almeno di 25 mt)

        self.images_paths = [p for p in self.database_paths]        # tutti i path delle immagini nel dataset
        self.images_paths += [p for p in self.queries_paths]        # più i path delle immagini delle query
        
        self.database_num = len(self.database_paths)                # restituisce il numero di paths (di immagini) del database
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]                       # prende il path dato l'index
        pil_img = open_image(image_path)                            # apre l'immagine con PIL e la restituisce in RGB
        normalized_img = self.base_transform(pil_img)               # applica le trasformazioni
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"        # restiuisce info sul database sottoforma di stringa
    
    def get_positives(self):
        return self.positives_per_query                              # ritorna la lista di positvi per ogni query (la query è l'indice)



##### GEOWARP TEST ####

class GeoTestDataset(data.Dataset):     
    def __init__(self, datasets_folder="datasets", dataset_name="pitts30k", split="test", positive_dist_threshold=25):
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
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # stessa mean e std del train
        ])

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_image = open_image(image_path)
        tensor_image = self.base_transform(pil_image)
        return tensor_image, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #gallery: {self.gallery_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):                     # ritorna i positives per query
        return self.soft_positives_per_query
