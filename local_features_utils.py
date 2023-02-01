import torch
import logging
import numpy as np


def CalculateKeypointCenters(boxes):
   # Helper function to compute feature centers, from RF boxes.
    # print(boxes)
    # x = boxes[:, :2]
    # y = boxes[:, 2:]
    # print(y)
    return torch.divide(torch.add(boxes[:, :2], boxes[:, 2:]),2.0)


def CalculateReceptiveBoxes(height, width, rf, stride, padding):
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height))
    coordinates = torch.reshape(torch.stack([x, y], axis=2), [-1, 2])               #  ho girato x e y rispetto alla repo per fare uscire gli stessi valori
    # [y,x,y,x]
    point_boxes = torch.tensor(torch.concat([coordinates, coordinates], 1), dtype=torch.float32)        

    # point_boxes pare avere tutte le combinazioni possibili di coordinate che prevedono [y, x e y, x]    # anche se noi abbiamo messo x,y
    # ricontrollare in caso di errori
    bias = torch.tensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes             # checked con i valori della repo

    # rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
    # Each box is represented by [ymin, xmin, ymax, xmax].

def extract_local_features():
    extracted_local_features = {
    'local_features': {
        'locations': np.array([]),
        'descriptors': np.array([]),
        'scales': np.array([]),
        'attention': np.array([]),
            }
    }

    rf, stride, padding = [211.0, 16.0, 105.0]

    feature_map = torch.rand([1, 64, 32, 32])
    attention_prob = torch.rand([1, 1, 32, 32])

    #eventuale scaling dell'immagine
    attention_prob = attention_prob.squeeze(0)         
    feature_map = feature_map.squeeze(0)

    rf_boxes = CalculateReceptiveBoxes(feature_map.shape[1], feature_map.shape[2], rf, stride, padding)

    attention_prob = attention_prob.view(-1)
    feature_map = feature_map.view(-1, feature_map.shape[0])          

    abs_thres = 0.5

    indices = (attention_prob >= abs_thres).nonzero().squeeze(1) 
    scale = 1

    selected_boxes = torch.index_select(rf_boxes, 0, indices)
    selected_features = torch.index_select(feature_map, 0, indices)
    selected_scores = torch.index_select(attention_prob, 0, indices)
    scales = torch.ones_like(selected_scores, dtype=torch.float32) / scale                 # dalla repo. Tensore di 1  riscalati rispetto alla scala

    # print(scales)

    # qua dovrebbe esserci il calcolo per la nuova scala (ma a sto punto credo che basterebbe richiamare il pezzo precedente)
    # a questo punto abbiamo un concatenamento dei risultati a scale diverse, una roba del genere:
    
    #     output_boxes = tf.concat([output_boxes, boxes], 0)
    # output_local_descriptors = tf.concat(
    #     [output_local_descriptors, local_descriptors], 0)
    # output_scales = tf.concat([output_scales, scales], 0)
    # output_scores = tf.concat([output_scores, scores], 0)

    # a qquesto punto salva le feature locali in una struttura dati chiamata BoxList, utilizzata tipicamente per object detection
    # ed esegue una non max suppression usando iou (che è 1 di default) e il numero di features salvate che ha l'unico di scopo di
    # eseguire un'altra thresold per rimuovere le features trovate nel caso in cui queste superino le mille unità

    # a questo punto occorerebbe calcolare i keypoints seguendo la pipeline

    locations = CalculateKeypointCenters(selected_boxes)
    print(locations)




# extract_local_features()