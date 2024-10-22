
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()                                                        # si mette il modello in evaluation mode
    with torch.no_grad():                                                       # all'interno del ciclo, il gradient è disabilitato (requires_grad=False)
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))                       # subset del dataset da valutare non considerando le immagini di query
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))    # creazione del dataloader in grado di iterare sul dataset
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")                           # ritorna un vettore non inizializzato con una riga per ogni sample da valutare
        for images, indices in tqdm(database_dataloader, ncols=100):                                              # è un numero di colonne pari alla dimensione di descrittori
            descriptors = model(images.to(args.device))                                                           # mette le immagini su device e ne calcola il risultato del MODELLO -> i descrittori
            descriptors = descriptors.cpu().numpy()                                                               # porta i descrittori su cpu e li traforma da tensori ad array
            all_descriptors[indices.numpy(), :] = descriptors                                                     # riempie l'array mettendo ad ogni indice il descrittore calcolato
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1                                                                              # sembra che venga valutata un'immagine per volta
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))    # in questo caso, crea un subset con sole query
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))            # crea il dataloader associato a questo secondo subset
        for images, indices in tqdm(queries_dataloader, ncols=100):                            
            descriptors = model(images.to(args.device))                       # fa lo stesso lavoro precedente, calcolando per ogni immagine di query il descrittore
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors                 # rimepiendo il vettore all_descriptors 
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]              # divide i descrittori delle queries
    database_descriptors = all_descriptors[:eval_ds.database_num]             # dai descrittori del database di immagini da classificare
    
    # Use a kNN to find predictions     ----    faiss (Facebook AI Similarity Search) è una libreria di Facebook che permette di effetuare una ricerca tra somiglianze in maniera efficiente
                                                             # faiss.IndexFlatL2 misura la l2 distance (o distanza euclidea) tra tutti i vettori dati e il quey vector 
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)      # qui sembra che lo stia inizializzando con la dimensione dei descrittori  
    faiss_index.add(database_descriptors)                    # dopodiché ci aggiunge tutti i descrittori delle immagini di test 
    del database_descriptors, all_descriptors                # elimina roba non piiù utile
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))   # effettua la ricerca con i descrittori delle query con i valori di recall specificati
                                                                        # questa parte quindi è svolta unicamente da questa libreria, che calcola la distanza euclidea (quindi la vicinanza)
                                                                        # per ogni k (preso da RECALL_VALUES) immagini con le immagini di query. Più k è alto è più ho possibilità di prendere la 
                                                                        # più vicina (lo si vede dopo)
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()               # per ogni query, restituisce l'immagine reale del dataset più vicina (credo, devo ancora guardare test_dataset)
    recalls = np.zeros(len(RECALL_VALUES))                      # vettore di recalls inizializzato a zero
    for query_index, preds in enumerate(predictions):           # per ogni predizione, prende indice e relativa predizione
        for i, n in enumerate(RECALL_VALUES):                   # per ogni valore delle recall values (sono 5 valori)
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):    # controlla che ogni valore nel primo 1Darray (quindi penso descrittore, non immagine) sia contenuto 
                                                                                # nel secondo. Quindi per ogni n controlla se le predizioni fino ad n (le n più vicine) contengono 
                                                                                # la relativa immagine di query (np.any -> almeno 1)
                recalls[i:] += 1                                                # se si, aumenta la relativa recall
                break                                                           # ed esce perché tanto l'ha già trovata. Quindi si favoriscono recall più basse
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100                               # valori di recall espressi in percentuale (cioè quante query in percentuale sono cadute in quel valore di recall)   
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])     # valori di recall in stringa
    return recalls, recalls_str

base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # stessa mean e std del train
])

def open_image(path):
    return Image.open(path).convert("RGB")
