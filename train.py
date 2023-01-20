
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import parser
import commons
import cosface_loss
import arcface_loss
import sphereface_loss 
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

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

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)      # istanzia il modello con backbone e dimensione del descrittore
                                                                           # passati da linea di comando
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")  # conta GPUs e CPUs

if args.resume_model is not None:                              # se c'è un modello pre-salvato da caricare
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)           # carica un oggetto salvato con torch.save(). Serve per deserializzare l'oggetto
    model.load_state_dict(model_state_dict)                    # copia parametri e buffer dallo state_dict all'interno del modello e nei suoi discendenti

                        # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor
                        # solo i layer con parametri hanno delle entry dentro al dizionario 

model = model.to(args.device).train()      # sposta il modello sulla GPU e lo mette in modalità training (alcuni layer si comporteranno di conseguenza)

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # utilizza l'algoritmo Adam per l'ottimizzazione

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

# Each group has its own classifier, which depends on the number of classes in the group (più gruppi ci sono, più classificatori sono usati con rispettivi optimizer)
# Noi abbiamo un solo gruppo perciò avremo un solo classifier

# Each group has its own classifier, which depends on the number of classes in the group
    # """Implement of large margin cosine distance:
    # Args:
    #     in_features: size of each input sample
    #     out_features: size of each output sample
    #     s: norm of input feature
    #     m: margin
    # """

logging.info(f"Using {args.loss} function") # dentro args.loss ho la mia loss: per settarla scrivere negli args --loss_function name quando fate partire il train
if args.loss == "cosface":
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]   # il classifier è dato dalla loss(dimensione descrittore, numero di classi nel gruppo) 
elif args.loss == "arcface": 
        classifiers = [arcface_loss.ArcFace(args.fc_output_dim, len(group)) for group in groups]
elif args.loss == "sphereface":
        classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
else:
    raise ValueError()

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers] # rispettivo optimizer

logging.info(f"Using {len(groups)} groups")                                                                                        # numero di gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")             # numero di classi nei gruppi
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")  # numero di immagini

# per capire gli output su, bisogna capire come sono state implementate le classi dei dataset

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold) 
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Resume
if args.resume_train:        # se è passato il path del checkpoint di cui fare il resume. E' come se salvasse un certo punto del train specifico (checkpoint)  
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)           # carica il checkpoint
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:                           # se non c'è resume, riparte da zero
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
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

for epoch_num in range(start_epoch_num, args.epochs_num):        # inizia il training
    
    #### Train
    epoch_start_time = datetime.now()                             # prende tempo e data di oggi
    # Select classifier and dataloader according to epoch            nell'idea di avere a che fare con più gruppi e più classifier
    current_group_num = epoch_num % args.groups_num               # avendo un solo gruppo, il resto è sempre zero. Se avessi due gruppi, nelle
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)       # sposta il classfier del gruppo nel device
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)           # sposta l'optimizer del gruppo nel device
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,     # il dataloader permetteva di iterare sul dataset
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)         # prende l'iteratore del dataloader
    model = model.train()                          # mette il modello in modalità training (non l'aveva già fatto?)
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)                      # 0 righe, 1 colonna -> l'array è vuoto
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):    # ncols è la grandezza della barra  
        images, targets, _ = next(dataloader_iterator)                     # ritorna il batch di immagini e le rispettive classi
        images, targets = images.to(args.device), targets.to(args.device)  # mette tutto su device
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)                              # se il device è cuda, fa questa augmentation
        
        model_optimizer.zero_grad()                                        # setta il gradiente a zero per evitare double counting (passaggio classico dopo ogni iterazione)
        classifiers_optimizers[current_group_num].zero_grad()              # fa la stessa cosa con l'ottimizzatore
        
        if not args.use_amp16:
            descriptors = model(images)                                     # inserisce il batch di immagini e restituisce il descrittore
            output = classifiers[current_group_num](descriptors, targets)   # riporta l'output del classifier (applica quindi la loss ai batches). Però passa sia descrittore cha label
            loss = criterion(output, targets)                               # calcola la loss (in funzione di output e target)
            loss.backward()                                                 # calcola il gradiente per ogni parametro che ha il grad settato a True
            epoch_losses = np.append(epoch_losses, loss.item())             # in epoch losses ci appende questa loss
            del loss, output, images                                        # elimina questi oggetti. Con la keyword del, l'intento è più chiaro
            model_optimizer.step()                                          # update dei parametri insieriti nell'ottimizzatore del modello
            classifiers_optimizers[current_group_num].step()                # update anche dei parametri del layer classificatore 
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():                                 # funzionamento che sfrutta amp16 per uno speed-up. Non trattato
                descriptors = model(images)                                 # comunque di base sono gli stessi passaggi ma con qualche differenza  
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
                  f"loss = {epoch_losses.mean():.4f}")                  # stampa la loss

    ## Se si vuole fare un grafico, si può usare "epoch_losses"

    #### Evaluation
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

# Fa un checkpoint ad ogni epoca salvando il dizionario di su (chiamato state in save_checkpoint) ed inoltre salva anche il modello
# finora migliore come "best_model". Questo significa che non è detto che il migliore sia nella ultima epoca. Anche perché ad ogni epoca 
# il gruppo cambia

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")    # carica il best model (salvato da save_checkpoint se is_best è True)
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)                   # prova il modello migliore sul dataset di test (queries v1)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")
