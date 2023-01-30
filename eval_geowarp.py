
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

##### MODEL #####
features_extractor = network.FeatureExtractor(args.backbone, args.fc_output_dim) 
global_features_dim = commons.get_output_dim(features_extractor, "gem")    
homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1) # inizializza il layer homography

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_fe is not None:
    state_dict = torch.load(args.resume_fe)
    features_extractor.load_state_dict(state_dict)
    del state_dict
else:
    logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                    "Feature Extractor is not initialized!")

if args.resume_hr is not None:
    state_dict = torch.load(args.resume_hr)
    homography_regression.load_state_dict(state_dict)
    del state_dict
else:
    logging.warning("WARNING: --resume_hr is set to None, meaning that the "
                    "Homography Regression is not initialized!")

model = network.GeoWarp(features_extractor, homography_regression)          # mette il modello in evaluation 
model = model.to(args.device)

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold)

logging.info(f"Start testing")
recalls, recalls_str, predictions = test.test_geowarp(args, test_ds, model)                   # prova il modello migliore sul dataset di test (queries v1)

logging.info(f"Start re-ranking")
_, reranked_recalls_str = test.test_reranked(args, model, predictions, test_ds, num_reranked_predictions = args.num_reranked_preds) # num_reranked_predictions, di default sono 5

logging.info(f"Test without warping: {test_ds}: {recalls_str}")
logging.info(f"  Test after warping: {test_ds}: {reranked_recalls_str}") # stampa le recall warpate

logging.info("Experiment finished (without any errors)")
