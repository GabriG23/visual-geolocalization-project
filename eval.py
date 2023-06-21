
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test_reranked
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True 
args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.fm_reduction_dim, args.reduction)           

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)                       
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)

test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)

recalls, recalls_str, reranked_recalls, reranked_recalls_str = test_reranked.test(args, test_ds, model)  
logging.info(f"{test_ds}: Recalls :\t\t {recalls_str}")
logging.info(f"{test_ds}: Reranked Recalls : {reranked_recalls_str}")
