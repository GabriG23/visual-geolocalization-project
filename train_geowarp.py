
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from torchvision.transforms.functional import hflip

import test
import util
import parser
import commons
import cosface_loss
import arcface_loss
import sphereface_loss 
import augmentations
from model import network
from datasets.warping_dataset import HomographyDataset
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.prediction_dataset import DatasetQP

torch.backends.cudnn.benchmark = True  # Provides a speedup
                                        # se il modello non cambia e l'input size rimane lo stesso, si può beneficiare
                                        # mettendolo a true
args = parser.parse_arguments()         # prende gli argomenti del comando
start_time = datetime.now() # starting time
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"   # definisce la cartella in cui saranno salvati i log
commons.make_deterministic(args.seed)                       # rende il risultato deterministico. Se seed == -1 è non deterministico
commons.setup_logging(output_folder, console="debug")       # roba che riguarda i file di log, vedi commons per ulteriori dettagli
logging.info(" ".join(sys.argv))                            # sys.argv è la lista di stringhe degli argomenti di linea di comando 
logging.info(f"Arguments: {args}")                          # questi sono gli argomenti effettivi
logging.info(f"The outputs are being saved in {output_folder}")

##### MODEL #####
features_extractor = network.FeatureExtractor(args.backbone, args.fc_output_dim)  # arch alexnet, vgg16 o resnet50, pooling netvlad o gem, args.arch, args.pooling
global_features_dim = commons.get_output_dim(features_extractor, "gem")              # dimensione dei descrittori = 512 con resnet18

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")  # conta GPUs e CPUs
if args.resume_model is not None:                              # se c'è un modello pre-salvato da caricare
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)           # carica un oggetto salvato con torch.save(). Serve per deserializzare l'oggetto
    features_extractor.load_state_dict(model_state_dict)                    # copia parametri e buffer dallo state_dict all'interno del modello e nei suoi discendenti

homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1) # inizializza il layer homography

model = network.GeoWarp(features_extractor, homography_regression).cuda().eval()
#model = torch.nn.DataParallel(model)        # parallelizes the application by splitting the input across the specified devices by chunking in the batch dimenesion
                                            # in the forward pass, the module is replicated on each device, and each replica handles a portion of the input.
                                            # During the backward pass, gradients from each replica are summed into the original module

#model = model.to(args.device).train()       # sposta il modello sulla GPU e lo mette in modalità training (alcuni layer si comporteranno di conseguenza)
##### MODEL #####

##### DATASETS & DATALOADERS ######

# dataset per il training con i gruppi
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]  # gruppi cosplace

# dataset per il warping (uno per gruppo)
ss_dataset = [HomographyDataset(args, args.train_set_folder, M=args.M, N=args.N, current_group=n, min_images_per_class=args.min_images_per_class, k=args.k) for n in range(args.groups_num)] # k = parameter k, defining the difficulty of ss training data, default = 0.6    

# dataset per le prediction(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold) 
if args.consistency_w != 0 or args.features_wise_w != 0:   # le prediction le calcola solo per le 2 loss, quindi se non ce le ho, non mi serve prenderle dal DB anche perchè non ho le query di train
    dataset_qp = [DatasetQP(model, global_features_dim, group, qp_threshold=args.qp_threshold) for group in groups]   # threshold = 1.2 di default
    dataloader_qp = commons.InfiniteDataLoader(dataset_qp, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)  # num_worker si potrebbe abbassare, ora è 8
    data_iter_qp = iter(dataloader_qp)   # iteratore prediction

# Validation and Test Dataset
val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold) 
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold)

logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")
##### DATASETS & DATALOADERS ######

##### LOSS & OPTIMIZER #####
criterion = torch.nn.CrossEntropyLoss()  # criterio usato per CosPlace. Abbiamo un problema di Classificazione
mse = torch.nn.MSELoss()  # criterio usato da GeoWarp. MSE misura the mean squared error tra gli elementi in input x e il target y. Qui abbiamo un problema di Regressione

model_optimizer = torch.optim.Adam(features_extractor.parameters(), lr=args.lr)  # utilizza l'algoritmo Adam per l'ottimizzazione
optim = torch.optim.Adam(homography_regression.parameters(), lr=args.lr) # anche in cosplace usa adam

logging.info(f"Using {args.loss_function} function") # dentro args.loss ho la mia loss: per settarla scrivere negli args --loss_function name quando fate partire il train
if args.loss_function == "cosface":
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]   # il classifier è dato dalla loss(dimensione descrittore, numero di classi nel gruppo) 
elif args.loss_function == "arcface": 
        classifiers = [arcface_loss.ArcFace(args.fc_output_dim, len(group)) for group in groups]
elif args.loss_function == "sphereface":
        classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
else:
    raise ValueError()

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers] # rispettivo optimizer

logging.info(f"Using {len(groups)} groups")                                                                                        # numero di gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")             # numero di classi nei gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")  # numero di immagini
##### LOSS & OPTIMIZER #####

#### Resume #####
if args.resume_train:        # se è passato il path del checkpoint di cui fare il resume. E' come se salvasse un certo punto del train specifico (checkpoint)  
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)           # carica il checkpoint
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:                           # se non c'è resume, riparte da zero
    best_val_recall1 = start_epoch_num = 0
#### Resume #####

##### Train / evaluation loop #####
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
            f"each epoch has {args.iterations_per_epoch} iterations " +
            f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
            f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

if args.augmentation_device == "cuda":           # data augmentation. Da cpu a gpu cambia solo il tipo di crop
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                        scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()
##### Train / evaluation loop #####

##### TRAIN #####
for epoch_num in range(start_epoch_num, args.epochs_num):        # inizia il training
    #### Train
    epoch_start_time = datetime.now()                             # prende tempo e data di oggi
    current_group_num = epoch_num % args.groups_num               # avendo un solo gruppo, il resto è sempre zero. Se avessi due gruppi, nelle
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)       # sposta il classfier del gruppo nel device
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)           # sposta l'optimizer del gruppo nel device
    
    # dataloader per cosplace, permetteva di iterare sul dataset, batch size = 32
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=True)
    dataloader_iterator = iter(dataloader)         # prende l'iteratore del dataloader
    # dataloader per il warping dataset
    ss_dataloader = commons.InfiniteDataLoader(ss_dataset[current_group_num], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=True)
    ss_data_iter = iter(ss_dataloader)   # crea iteratore sui data loader

    model = model.train()       # mette il modello in modalità training (non l'aveva già fatto?)  
    # epoch_losses = np.zeros((0, 4), dtype=np.float32)                      # inizializza il vettore delle loss, ne abbiamo 4, cosface e le altre 3
    epoch_losses = np.zeros((0, 2), dtype=np.float32)    
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):    # ncols è la grandezza della barra, 10k iterazioni per gruppo
        
        images, targets, _ = next(dataloader_iterator)                     # ritorna il batch di immagini e le rispettive classi (target)
        images, targets = images.to(args.device), targets.to(args.device)  # mette tutto su device (cuda o cpu)

        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)      # se il device è cuda, fa questa augmentation SULL'INTERO BATCH, se siamo sulla cpu, applica le trasformazioni ad un'immagine per volta, direttamente in train_dataset
        # dal warping dataset prende le due immagini warped e i due punti delle intersezioni
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = next(ss_data_iter)
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = warped_img_1.to(args.device), warped_img_2.to(args.device), warped_intersection_points_1.to(args.device), warped_intersection_points_2.to(args.device)  # warping dataset
        # Iq, Ip, tq e tp
        # if args.consistency_w != 0 or args.features_wise_w != 0: #entra solo se devo calcolare le due loss
        #     queries, positives = next(data_iter_qp)
        #     queries, positives = queries.to(args.device), positives.to(args.device)

        with torch.no_grad():                          # no gradient
            # fq e fp, calcola la misura di similarity
            similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])       # crea le due matrici similarity
            # if args.consistency_w != 0:
            #     queries_cons = queries[:args.batch_size]                                                 # prende la conistency = 16                                          
            #     positives_cons = positives[:args.batch_size]                                             # anche qui = 16
            #     similarity_matrix_q2p, similarity_matrix_p2q = model("similarity", [queries_cons, positives_cons])   # seconde similarity matrix
            #     fl_similarity_matrix_q2p, fl_similarity_matrix_p2q = model("similarity", [hflip(queries_cons), hflip(positives_cons)])
            #     del queries_cons, positives_cons                                                                     # le cancella non ci servono più
    
        model_optimizer.zero_grad()                                        # setta il gradiente a zero per evitare double counting (passaggio classico dopo ogni iterazione)
        classifiers_optimizers[current_group_num].zero_grad()              # fa la stessa cosa con l'ottimizzatore
        optim.zero_grad()

        if not args.use_amp16:
            descriptors = model("features_extractor", [images, "global"])                                     # inserisce il batch di immagini e restituisce il descrittore
            output = classifiers[current_group_num](descriptors, targets)   # riporta l'output del classifier (applica quindi la loss ai batches). Però passa sia descrittore cha label
            loss = criterion(output, targets)                               # calcola la loss (in funzione di output e target)
            # loss *= args.loss_weight # moltiplichiamo per un peso, per ora è 1
            loss.backward()                                                 # calcola il gradiente per ogni parametro che ha il grad settato a True
            loss = loss.item()
            #epoch_losses = np.append(epoch_losses, loss.item())             # in epoch losses ci appende questa loss
            del output, images                                        # elimina questi oggetti. Con la keyword del, l'intento è più chiaro
                            
            if args.ss_w != 0:  # ss_loss  # self supervised loss    guides the network to learn to estimate the points 
                pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
                pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
                ss_loss = (mse(pred_warped_intersection_points_1[:, :4].float(), warped_intersection_points_1.float())+
                        mse(pred_warped_intersection_points_1[:, 4:].float(), warped_intersection_points_2.float()) +
                        mse(pred_warped_intersection_points_2[:, :4].float(), warped_intersection_points_2.float()) +
                        mse(pred_warped_intersection_points_2[:, 4:].float(), warped_intersection_points_1.float()))
                ss_loss.backward()
                ss_loss = ss_loss.item()
                del pred_warped_intersection_points_1, pred_warped_intersection_points_2
            else:
                ss_loss = 0
            
            # consistency_loss    # genera delle pseudo label come ground truth to further improve robustness
            # if args.consistency_w != 0:
            #     pred_intersection_points_q2p = model("regression", similarity_matrix_q2p)
            #     pred_intersection_points_p2q = model("regression", similarity_matrix_p2q)
            #     fl_pred_intersection_points_q2p = model("regression", fl_similarity_matrix_q2p)
            #     fl_pred_intersection_points_p2q = model("regression", fl_similarity_matrix_p2q)

            #     new_points = torch.cat((fl_pred_intersection_points_q2p[:, 4:], fl_pred_intersection_points_q2p[:, :4]), 1)
            #     third_points = torch.zeros_like(new_points)
            #     third_points[:, 0::2, :] = new_points[:, 1::2, :]
            #     third_points[:, 1::2, :] = new_points[:, 0::2, :]
            #     third_points[:, :, 0] *= -1

            #     fourth_points = torch.zeros_like(fl_pred_intersection_points_p2q)
            #     fourth_points[:, 0::2, :] = fl_pred_intersection_points_p2q[:, 1::2, :]
            #     fourth_points[:, 1::2, :] = fl_pred_intersection_points_p2q[:, 0::2, :]
            #     fourth_points[:, :, 0] *= -1

            #     four_predicted_points = [
            #         torch.cat((pred_intersection_points_q2p[:, 4:], pred_intersection_points_q2p[:, :4]), 1),
            #         pred_intersection_points_p2q,
            #         third_points,
            #         fourth_points
            #     ]
            #     four_predicted_points_centroids = torch.cat([p[None] for p in four_predicted_points]).mean(0).detach()
            #     consistency_loss = sum([mse(pred, four_predicted_points_centroids) for pred in four_predicted_points])
            #     consistency_loss *= args.consistency_w
            #     consistency_loss.backward()
            #     consistency_loss = consistency_loss.item()
            #     del pred_intersection_points_q2p, pred_intersection_points_p2q
            #     del fl_pred_intersection_points_q2p, fl_pred_intersection_points_p2q
            #     del four_predicted_points
            # else:
            #     consistency_loss = 0
        
            # if args.features_wise_w != 0: # features_wise_loss     # assicura che le feature siano più vicine possibile dopo che vengono estratte dal training
            #     queries_fw = queries[:args.batch_size_features_wise]
            #     positives_fw = positives[:args.batch_size_features_wise]
            #     # Add random weights to avoid numerical instability
            #     random_weights = (torch.rand(args.batch_size_features_wise, 4)**0.1).cuda()
            #     w_queries, w_positives, _, _ = datasets.warping_dataset.compute_warping(model, queries_fw, positives_fw, weights=random_weights)
            #     f_queries = model("features_extractor", [w_queries, "local"])
            #     f_positives = model("features_extractor", [w_positives, "local"])
            #     features_wise_loss = mse(f_queries, f_positives)
            #     features_wise_loss *= args.features_wise_w
            #     features_wise_loss.backward()
            #     features_wise_loss = features_wise_loss.item()

            #     del queries, positives, queries_fw, positives_fw, w_queries, w_positives, f_queries, f_positives
            # else:
            #     features_wise_loss = 0
            
            # epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss, consistency_loss, features_wise_loss]]))) # concateniamo le loss
            # del loss, ss_loss, consistency_loss, features_wise_loss
            epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]]))) # concateniamo le loss
            del loss, ss_loss
            model_optimizer.step()                                          # update dei parametri insieriti nell'ottimizzatore del modello
            classifiers_optimizers[current_group_num].step()                # update anche dei parametri del layer classificatore 
            optim.step()
        else:  # Use AMP 16
            # a me non dovrebbe servire
            with torch.cuda.amp.autocast():                                    # funzionamento che sfrutta amp16 per uno speed-up. Non trattato
                descriptors = model("feature_extractor", [images, "global"])   # comunque di base sono gli stessi passaggi ma con qualche differenza  
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
        
            # epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss, consistency_loss, features_wise_loss]]))) # concateniamo le loss
            # del loss, ss_loss, consistency_loss, features_wise_loss
        
            epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]]))) # concateniamo le loss
            del loss, ss_loss

    classifiers[current_group_num] = classifiers[current_group_num].cpu()   # passsa il classifier alla cpu termina l'epoca  
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")   # passa anche l'optimizer alla cpu

    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"loss = {epoch_losses.mean():.4f}")                  # stampa la loss

    ##### EVALUATION #####

    recalls, recalls_str, _ = test.test_geowarp(args, val_ds, model)              # passa validation dataset e modello (allenato) per il calcolo delle recall
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1                            # lo confronta con il valore della recall maggiore. E' un valore booleano
    best_val_recall1 = max(recalls[0], best_val_recall1)               # prende il valore massimo tra le due  
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)

    ##### EVALUATION #####
##### TRAIN #####

##### TEST #####

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")    # carica il best model (salvato da save_checkpoint se is_best è True)
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")

recalls, recalls_str, predictions = test.test_geowarp(args, test_ds, model)                   # prova il modello migliore sul dataset di test (queries v1)

logging.info(f"Normal test: {test_ds}: {recalls_str}")

_, reranked_recalls_str = test.test_reranked(model, predictions, test_ds, num_reranked_predictions = args.num_reranked_preds) # num_reranked_predictions, di default sono 5

logging.info(f"Test after warping - {reranked_recalls_str}") # stampa le recall warpate

logging.info("Experiment finished (without any errors)")