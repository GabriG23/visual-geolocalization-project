
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
from local_features_utils import retrieve_locations_and_descriptors, match_features

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:      
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()                                        
    with torch.no_grad():                                      
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))        
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))  
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")     
        all_local_descriptors = np.empty((len(eval_ds), args.fm_reduction_dim, 14, 14), dtype="float32")
        all_att_prob = np.empty((len(eval_ds), 1, 14, 14), dtype="float32")
        
        
        for images, indices in tqdm(database_dataloader, ncols=100):                       
            global_descriptors, _, _, _, local_descriptors, attn_scores = model(images.to(args.device))                                     
            global_descriptors = global_descriptors.cpu().numpy()                                       
            local_descriptors = local_descriptors.cpu().numpy() 
            attn_scores = attn_scores.cpu().numpy() 
            all_descriptors[indices.numpy(), :] = global_descriptors                               
            all_local_descriptors[indices.numpy(), :] = local_descriptors
            all_att_prob[indices.numpy(), :] = attn_scores


        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1                                                        
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))      
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))        
        for images, indices in tqdm(queries_dataloader, ncols=100):
            global_descriptors, _, _, _, local_descriptors, attn_scores = model(images.to(args.device))           
            global_descriptors = global_descriptors.cpu().numpy()
            local_descriptors = local_descriptors.cpu().numpy() 
            attn_scores = attn_scores.cpu().numpy() 
            all_descriptors[indices.numpy(), :] = global_descriptors                             
            all_local_descriptors[indices.numpy(), :] = local_descriptors
            all_att_prob[indices.numpy(), :] = attn_scores      

    queries_global_descriptors = all_descriptors[eval_ds.database_num:]   
    queries_local_descriptors = all_local_descriptors[eval_ds.database_num:]
    queries_att_prob = all_att_prob[eval_ds.database_num:]

    database_global_descriptors = all_descriptors[:eval_ds.database_num]   
    database_local_descriptors = all_local_descriptors[:eval_ds.database_num]
    database_att_prob = all_att_prob[:eval_ds.database_num]
                                                            
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)    
    faiss_index.add(database_global_descriptors)                 
    del all_local_descriptors, all_descriptors, all_att_prob          
    
   
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_global_descriptors, max(RECALL_VALUES))    
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))    
    reranked_recalls = np.zeros(len(RECALL_VALUES))                
    for query_index, preds in enumerate(tqdm(predictions, ncols=100)):  
        reranked_preds = RerankByGeometricVerification(preds, distances[query_index], queries_local_descriptors[query_index], 
                                    queries_att_prob[query_index], database_local_descriptors[preds], database_att_prob[preds])
        for i, n in enumerate(RECALL_VALUES):           
            if np.any(np.in1d(reranked_preds[:n], positives_per_query[query_index])):                                                                                                                         
                reranked_recalls[i:] += 1                                               
                break     
        for i, n in enumerate(RECALL_VALUES):            
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):                                                                                                                                          
                recalls[i:] += 1                                             
                break                                                           

    recalls = recalls / eval_ds.queries_num * 100   
    reranked_recalls = reranked_recalls / eval_ds.queries_num * 100                            

    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])  
    reranked_recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, reranked_recalls)])    
    return recalls, recalls_str, reranked_recalls, reranked_recalls_str


def RerankByGeometricVerification(query_predictions, distances, query_descriptors, query_attention_prob, 
                    images_local_descriptors, images_attention_prob):
   
    query_locations, query_descriptors = retrieve_locations_and_descriptors(query_attention_prob, query_descriptors)

    inliers_and_initial_scores = []                   
    for i, preds in enumerate(query_predictions):

        image_locations, image_descriptors = retrieve_locations_and_descriptors(images_attention_prob[i], images_local_descriptors[i])

        match_result = match_features(
            query_locations,  
            query_descriptors,
            image_locations,
            image_descriptors,
            use_ratio_test=False,
            RANSAC=True)

        inliers_and_initial_scores.append([preds, match_result, distances[i]])

    inliers_and_initial_scores = sorted(inliers_and_initial_scores, key=lambda x : (x[1], -x[2]), reverse=True)
    new_rank = [x[0] for x in inliers_and_initial_scores]

    return new_rank