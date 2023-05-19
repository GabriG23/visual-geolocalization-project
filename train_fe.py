
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
model = network.FeatureExtractor(args.backbone, args.fc_output_dim)     # arch alexnet, vgg16 o resnet50, pooling netvlad o gem, args.arch, args.pooling

features_extractor = model.to(args.device).train()      # sposta il modello sulla GPU e lo mette in modalità training (alcuni layer si comporteranno di conseguenza)

##### MODEL #####

##### DATASETS & DATALOADERS ######
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]  # gruppi cosplace

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)                                 # Validation and Test Dataset
test_ds = TestDataset(args.test_set_folder, queries_folder="queries", positive_dist_threshold=args.positive_dist_threshold)

logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")
##### DATASETS & DATALOADERS ######

##### LOSS & OPTIMIZER #####
criterion = torch.nn.CrossEntropyLoss()  # criterio usato per CosPlace. Abbiamo un problema di Classificazione

model_optimizer = torch.optim.Adam(features_extractor.parameters(), lr=args.lr)  # utilizza l'algoritmo Adam per l'ottimizzazione

classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]   # il classifier è dato dalla loss(dimensione descrittore, numero di classi nel gruppo) 
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
            augmentations.DeviceAgnosticRandomResizedCrop([224, 224],
                                                        scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()
##### Train / evaluation loop #####

##### TRAIN #####
for epoch_num in range(start_epoch_num, args.epochs_num):      #### Train
    
    epoch_start_time = datetime.now()                                                        # prende tempo e data di oggi
    current_group_num = epoch_num % args.groups_num                                          # avendo un solo gruppo, il resto è sempre zero. Se avessi due gruppi, nelle
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)          # sposta il classfier del gruppo nel device
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)              # sposta l'optimizer del gruppo nel device
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=True)     # dataloader per cosplace, permetteva di iterare sul dataset, batch size = 32
    dataloader_iterator = iter(dataloader)         # prende l'iteratore del dataloader sui gruppi


    model = model.train()       # mette il modello in modalità training (non l'aveva già fatto?)  
    epoch_losses = np.zeros((0, 1), dtype=np.float32)    

    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):    # ncols è la grandezza della barra, 10k iterazioni per gruppo
        
        images, targets, _ = next(dataloader_iterator)                     # ritorna il batch di immagini e le rispettive classi (target)
        images, targets = images.to(args.device), targets.to(args.device)  # mette tutto su device (cuda o cpu)

        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)      # se il device è cuda, fa questa augmentation SULL'INTERO BATCH, se siamo sulla cpu, applica le trasformazioni ad un'immagine per volta, direttamente in train_dataset

        model_optimizer.zero_grad()                                        # setta il gradiente a zero per evitare double counting (passaggio classico dopo ogni iterazione)
        classifiers_optimizers[current_group_num].zero_grad()              # fa la stessa cosa con l'ottimizzatore

        if not args.use_amp16:
            descriptors = model(images)                                     # inserisce il batch di immagini e restituisce il descrittore
            output = classifiers[current_group_num](descriptors, targets)   # riporta l'output del classifier (applica quindi la loss ai batches). Però passa sia descrittore cha label
            loss = criterion(output, targets)                               # calcola la loss (in funzione di output e target)
            loss.backward()                                                 # calcola il gradiente per ogni parametro che ha il grad settato a True
            epoch_losses = np.append(epoch_losses, loss.item())    # concateniamo le loss
            del output, images                                        # elimina questi oggetti. Con la keyword del, l'intento è più chiaro       
            model_optimizer.step()                                          # update dei parametri insieriti nell'ottimizzatore del modello
            classifiers_optimizers[current_group_num].step()                # update anche dei parametri del layer classificatore 
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():                                    # funzionamento che sfrutta amp16 per uno speed-up. Non trattato
                descriptors = model(images)     # comunque di base sono gli stessi passaggi ma con qualche differenza  
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()

    classifiers[current_group_num] = classifiers[current_group_num].cpu()   # passsa il classifier alla cpu termina l'epoca  
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")   # passa anche l'optimizer alla cpu

    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"loss = {epoch_losses.mean():.4f}")                       # stampa la loss

    ##### EVALUATION #####

    recalls, recalls_str = test.test(args, val_ds, model)              # passa validation dataset e modello (allenato) per il calcolo delle recall
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

logging.info(f"Start testing")
recalls, recalls_str = test.test(args, test_ds, model)                   # prova il modello migliore sul dataset di test (queries v1)

logging.info(f"Test without warping: {test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")


