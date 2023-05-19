
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

torch.backends.cudnn.benchmark = True  # Provides a speedup, se il modello non cambia e l'input size rimane lo stesso, si può beneficiare,  mettendolo a true

args = parser.parse_arguments()         # prende gli argomenti del comando
start_time = datetime.now() # starting time
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"   # definisce la cartella in cui saranno salvati i log
commons.make_deterministic(args.seed)                                                # rende il risultato deterministico. Se seed == -1 è non deterministico
commons.setup_logging(output_folder, console="debug")                                # roba che riguarda i file di log, vedi commons per ulteriori dettagli
logging.info(" ".join(sys.argv))                                                     # sys.argv è la lista di stringhe degli argomenti di linea di comando 
logging.info(f"Arguments: {args}")                                                    # questi sono gli argomenti effettivi
logging.info(f"The outputs are being saved in {output_folder}")

##### MODEL #####
features_extractor = network.FeatureExtractor(args.backbone, args.fc_output_dim)     # arch alexnet, vgg16 o resnet50, pooling netvlad o gem, args.arch, args.pooling
# global_features_dim = commons.get_output_dim(features_extractor, "gem")              # dimensione dei descrittori = 512 con resnet18

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")  # conta GPUs e CPUs
if args.resume_fe is not None:                                                    # se c'è un modello pre-salvato da caricare
    logging.debug(f"Loading model from {args.resume_fe}")
    model_state_dict = torch.load(args.resume_fe)                                 # carica un oggetto salvato con torch.save(). Serve per deserializzare l'oggetto
    features_extractor.load_state_dict(model_state_dict)                            # copia parametri e buffer dallo state_dict all'interno del modello e nei suoi discendenti

homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1) # inizializza il layer homography

model = network.GeoWarp(features_extractor, homography_regression)
model = model.to(args.device).eval()      # sposta il modello sulla GPU e lo mette in modalità training (alcuni layer si comporteranno di conseguenza)

##### MODEL #####

##### DATASETS & DATALOADERS ######
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]  # gruppi cosplace

ss_dataset = [HomographyDataset(args, args.train_set_folder, M=args.M, N=args.N, current_group=n, min_images_per_class=args.min_images_per_class, k=args.k) for n in range(args.groups_num)] # k = parameter k, defining the difficulty of ss training data, default = 0.6    

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold)

logging.info(f"Test set: {test_ds}")
##### DATASETS & DATALOADERS ######

##### LOSS & OPTIMIZER #####
criterion_mse = torch.nn.MSELoss()  # criterio usato da GeoWarp. MSE misura the mean squared error tra gli elementi in input x e il target y. Qui abbiamo un problema di Regressione

model_optimizer = torch.optim.Adam(homography_regression.parameters(), lr=args.lr)  # utilizza l'algoritmo Adam per l'ottimizzazione

logging.info(f"Using {len(groups)} groups")                                                                                        # numero di gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")             # numero di classi nei gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")  # numero di immagini
##### LOSS & OPTIMIZER #####

#### Resume #####
if args.resume_train:        # se è passato il path del checkpoint di cui fare il resume. E' come se salvasse un certo punto del train specifico (checkpoint)  
    model, homography_regression, model_optimizer, start_epoch_num = \
        util.resume_train_homography(args, output_folder, model, model_optimizer, homography_regression)           # carica il checkpoint
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
else:                           # se non c'è resume, riparte da zero
    start_epoch_num = 0
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
            augmentations.DeviceAgnosticRandomResizedCrop([224, 224],
                                                        scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

##### Train / evaluation loop #####

##### TRAIN #####
for epoch_num in range(start_epoch_num, args.epochs_num):      #### Train
    
    epoch_start_time = datetime.now()                                                  # prende tempo e data di oggi
    
    homography_regression = homography_regression.train() 
    epoch_losses = np.zeros((0, 1), dtype=np.float32)    
    
    current_group_num = epoch_num % args.groups_num                                          # avendo un solo gruppo, il resto è sempre zero. Se avessi due gruppi, nelle

    ss_dataloader = commons.InfiniteDataLoader(ss_dataset[current_group_num], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=True)
    ss_data_iter = iter(ss_dataloader)              # crea iteratore sui dati di warping

    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):    # ncols è la grandezza della barra, 10k iterazioni per gruppo
        
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = next(ss_data_iter)   # dal warping dataset prende le due immagini warped e i due punti delle intersezioni
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = warped_img_1.to(args.device), warped_img_2.to(args.device), warped_intersection_points_1.to(args.device), warped_intersection_points_2.to(args.device)  # warping dataset

        with torch.no_grad():  # no gradient
            similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])  # fq e fp, calcola la misura di similarity
    
        model_optimizer.zero_grad()                                        # setta il gradiente a zero per evitare double counting (passaggio classico dopo ogni iterazione)
                  
        if args.ss_w != 0:  # ss_loss  # self supervised loss    guides the network to learn to estimate the points 
            pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
            pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
            ss_loss = (criterion_mse(pred_warped_intersection_points_1[:, :4], warped_intersection_points_1)+
                    criterion_mse(pred_warped_intersection_points_1[:, 4:], warped_intersection_points_2) +
                    criterion_mse(pred_warped_intersection_points_2[:, :4], warped_intersection_points_2) +
                    criterion_mse(pred_warped_intersection_points_2[:, 4:], warped_intersection_points_1))
            ss_loss *= args.ss_w
            ss_loss.backward()
            epoch_losses = np.append(epoch_losses, ss_loss.item()) 
            del ss_loss, pred_warped_intersection_points_1, pred_warped_intersection_points_2
        else:
            ss_loss = 0
        
        model_optimizer.step()                                          # update dei parametri insieriti nell'ottimizzatore del modello

    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"loss = {epoch_losses.mean():.4f}")                       # stampa la loss

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint_homography({
        "epoch_num": epoch_num + 1,
        "homography_state_dict": homography_regression.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict()
    }, output_folder)
    ##### EVALUATION #####
##### TRAIN #####

##### TEST #####

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

logging.info(f"Start testing")

best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")    # carica il best model (salvato da save_checkpoint se is_best è True)
model.load_state_dict(best_model_state_dict)

recalls, recalls_str, predictions, _, _ = \
    test.compute_features(args, test_ds, model)

_, reranked_recalls_str = test.test_reranked(args, model, predictions, test_ds)

logging.info(f"Test without warping: {test_ds}: {recalls_str}")
logging.info(f"  Test after warping: {test_ds}: {reranked_recalls_str}") # stampa le recall warpate

logging.info("Experiment finished (without any errors)")