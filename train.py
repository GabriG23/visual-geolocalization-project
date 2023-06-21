
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test, test_reranked
import util
import parser
import commons
import cosface_loss
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True                      

args = parser.parse_arguments()      
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"  
commons.make_deterministic(args.seed)                      
commons.setup_logging(output_folder, console="debug")     
logging.info(" ".join(sys.argv))                        
logging.info(f"Arguments: {args}")                       
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.fm_reduction_dim, args.reduction)                                                                            
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")   

if args.resume_model is not None:                 
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)        
    model.load_state_dict(model_state_dict)            

        
model = model.to(args.device).train()    

#### Optimizer
criterion = torch.nn.CrossEntropyLoss() 
if args.reduction:
    criterion_MSE = torch.nn.MSELoss() 

if args.reduction:
    autoencoder_optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=args.lr)

backbone_parameters = [*model.backbone_after_3.parameters(), *model.layers_4.parameters(), *model.aggregation.parameters(), *model.attn_classifier.parameters(), *model.attention.parameters()]      
model_optimizer = torch.optim.Adam(backbone_parameters, lr=args.lr)   

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]       
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]    

logging.info(f"Using {len(groups)} groups")                                                                                       
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")            
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")  

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Resume
if args.resume_train:        
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)       
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:           
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":      
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



for epoch_num in range(start_epoch_num, args.epochs_num):          
    
    #### Train
    epoch_start_time = datetime.now()                              
    current_group_num = epoch_num % args.groups_num                
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)     
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)      
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,   
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)                 
    model = model.train()                                  
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):   
        images, targets, _ = next(dataloader_iterator)                      
        images, targets = images.to(args.device), targets.to(args.device)  
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)                              
                                                                                                                                         
        model_optimizer.zero_grad()                                   
        classifiers_optimizers[current_group_num].zero_grad()    
        if args.reduction:
            autoencoder_optimizer.zero_grad()        
        
        if not args.use_amp16:
            descriptors, attn_logits, feature_map, rec_feature_map, _, attn_scores = model(images) 
            output = classifiers[current_group_num](descriptors, targets)        
            
            global_loss = criterion(output, targets)                                         
            attn_loss = criterion(attn_logits, targets)
            if args.reduction:
                rec_loss = criterion_MSE(rec_feature_map, feature_map)  
                loss = global_loss + attn_loss + rec_loss 
            else:
                loss = global_loss + attn_loss                      
    
            loss.backward()
            model_optimizer.step() 
            classifiers_optimizers[current_group_num].step() 
            if args.reduction:
                autoencoder_optimizer.step()                                             

            epoch_losses = np.append(epoch_losses, loss.item()) 
            
            del loss, output, images                                                                              
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():                                           
                descriptors = model(images)                                            
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()         
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")   
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"loss = {epoch_losses.mean():.4f}, ")


    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)  
      
    # !!!!!! 
    # Here we're running the validation only on global features, because using local features
    # could exceed the Colab limits, which might result in the training not being completed
           
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1                    
    best_val_recall1 = max(recalls[0], best_val_recall1)         


    if args.reduction:
      util.save_checkpoint({                                          
          "epoch_num": epoch_num + 1,
          "model_state_dict": model.state_dict(),
          "autoencoder_optimizer_state_dict": autoencoder_optimizer.state_dict(),
          "optimizer_state_dict": model_optimizer.state_dict(),
          "classifiers_state_dict": [c.state_dict() for c in classifiers],
          "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
          "best_val_recall1": best_val_recall1
      }, is_best, output_folder)
    else:
      util.save_checkpoint({                                          
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
      }, is_best, output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")          
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")                         
recalls, recalls_str, reranked_recalls, reranked_recalls_str = test_reranked.test(args, test_ds, model)                          
logging.info(f"{test_ds}: Recalls /t: {recalls_str}")
logging.info(f"{test_ds}: Reranked Recalls /t: {reranked_recalls_str}")

logging.info("Experiment finished (without any errors)")
