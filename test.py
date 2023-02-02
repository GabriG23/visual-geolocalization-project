
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
from local_features_utils import retrieve_locations_descriptors, match_features

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:      # restituisce l'array con le recall e un stringa che riporta i valori
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()                                        # si mette il modello in evaluation mode
    with torch.no_grad():                                       # all'interno del ciclo, il gradient è disabilitato (requires_grad=False)
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))             # subset del dataset da valutare non considerando le immagini di query
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))  # creazione del dataloader in grado di iterare sul dataset
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")     # ritorna un vettore non inizializzato con una riga per ogni sample da valutare
    
        for images, indices in tqdm(database_dataloader, ncols=100):                        # e un numero di colonne pari alla dimensione di descrittori
            global_descriptors, _, _, _, _, _ = model(images.to(args.device))                                     # mette le immagini su device e ne calcola il risultato del MODELLO -> i descrittori
            global_descriptors = global_descriptors.cpu().numpy()                                         # porta i descrittori su cpu e li traforma da tensori ad array
            # per quanto riguarda i local descriptors, lui sembra fare una sogliatura, che eventualmente sarà aggiunta successivamente
            # local_descriptors = local_descriptors.cpu().numpy() 
            all_descriptors[indices.numpy(), :] = global_descriptors                               # riempie l'array mettendo ad ogni indice il descrittore calcolato
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1                                                        # sembra che venga valutata un'immagine per volta
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))        # in questo caso, crea un subset con sole query
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))        # crea il dataloader associato a questo secondo subset
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))             # fa lo stesso lavoro precedente, calcolando per ogni immagine di query il descrittore
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors       # rimepiendo il vettore all_descriptors 
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]    # divide i descrittori delle queries
    database_descriptors = all_descriptors[:eval_ds.database_num]   # dai descrittori del database di immagini da classificare
    
    # Use a kNN to find predictions     ----    faiss (Facebook AI Similarity Search) è una libreria di Facebook che permette di effetuare una ricerca tra somiglianze in maniera efficiente
                                                            # faiss.IndexFlatL2 misura la l2 distance (o distanza euclidea) tra tutti i vettori dati e il quey vector 
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)     # qui sembra che lo stia inizializzando con la dimensione dei descrittori  
    faiss_index.add(database_descriptors)                   # dopodiché ci aggiunge tutti i descrittori delle immagini di test 
    del database_descriptors, all_descriptors               # elimina roba non più utile
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))    # effettua la ricerca con i descrittori delle query con i valori di recall specificati
                                                            # questa parte quindi è svolta unicamente da questa libreria, che calcola la distanza euclidea (quindi la vicinanza)
                                                            # per ogni k (preso da RECALL_VALUES) immagini con le immagini di query. Più k è alto è più ho possibilità di prendere la 
                                                            # più vicina (lo si vede dopo)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()           # per ogni query, restituisce le immagini più vicine alla query di 25 mt
    recalls = np.zeros(len(RECALL_VALUES))                  # vettore di recalls iniziaizzato a zero
    for query_index, preds in enumerate(predictions):       # per ogni predizione, prende indice e relativa predizione
        for i, n in enumerate(RECALL_VALUES):               # per ogni valore delle recall values (sono 5 valori)
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):    # controlla che ogni valore nel primo 1Darray (quindi penso descrittore, non immagine) sia contenuto 
                                                                                # nel secondo. Quindi per ogni n controlla se le predizioni fino ad n (le n più vicine) contengono 
                                                                                # la relativa immagine di query (np.any -> almeno 1)
                recalls[i:] += 1                                                # se si, aumenta la relativa recall
                break                                                           # ed esce perché tanto l'ha già trovata. Quindi si favoriscono recall più basse
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100                               # valori di recall espressi in percentuale (cioè quante query in percentuale sono cadute in quel valore di recall)                                                                                     
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])    # valori di recall in stringa
    return recalls, recalls_str


# predictions sono indici in ordine di predizione e restituisce anche le distanze che possiamo usare come score
# usare il meno per ordinarli come mi serve


# occhio che le immagini di query hanno una dimensione diversa e questo può dare fastidio
# perciò va fatto quando si fa inferenza

images_matched = []
for i in predictions:
    images_matched.append(database_subset_ds[i])

def RerankByGeometricVerification(predictions, distances, all_query_descriptors, all_query_attention_prob, 
                        all_images_local_descriptors, all_images_attention_prob):
    # ranks_before_gv[i] = np.argsort(-similarities)      # tieni conto di questo!!!
    ransac_seed = 0
    descriptor_matching_threshold = 1.0
    ransac_residual_threshold = 20.0
    use_ratio_test = False


    # num_to_rerank = 100
    for query_index, preds in enumerate(predictions):
        print(f"Re-ranking: with query {query_index} out of {len(predictions)}")

        num_matched_images = len(preds)

        inliers_and_initial_scores = []                # in 0 avrà gli outliers, in 1 avrà gli scores (già calcolati)
        for i in range(num_matched_images):
            inliers_and_initial_scores.append([0, distances[query_index][i]])
        for i in range(num_matched_images):
            if i > 0 and i % 10 == 0:
                print(f"/tRe-ranking: {i} out of {num_matched_images}")
            
            database_image_index = preds[i]

            # Load index image features.
            query_locations, query_descriptors = retrieve_locations_descriptors(all_query_descriptors[query_index], 
                                                                        all_query_attention_prob[query_index])
            
            database_image_locations, database_image_descriptors = retrieve_locations_descriptors(all_query_descriptors[database_image_index], 
                                                                        all_query_attention_prob[database_image_index])

            inliers_and_initial_scores[database_image_index][0], _ = match_features(
                query_locations,
                query_descriptors,
                database_image_locations,
                database_image_descriptors,
                ransac_seed=ransac_seed,
                descriptor_matching_threshold=descriptor_matching_threshold,
                ransac_residual_threshold=ransac_residual_threshold,
                use_ratio_test=use_ratio_test)

            inliers_and_initial_scores = sorted(range(inliers_and_initial_scores), key=lambda x : (inliers_and_initial_scores[x][0], inliers_and_initial_scores[x][1]), reverse=True)

            # parte di ricalcolo della recall una volta ottenuti gli inliers



    # return output_reranks