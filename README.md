### Students
- Aiello Davide: s303296
- Gabriele Greco: s303435
# Visual Geolocalization: mapping images to GPS

This is the official repository for the Advanced Machine Learning Exam of 30/06/2023, project delivered on 22/06/2023.

All the train and test procedure are performed through Google Colaboratory environment. The following operations are used to run all the algorithm


## Libraries
```
!pip3 install 'torch>=1.8.2'
!pip3 install 'torchvision>=0.9.2'
!pip3 install 'faiss_cpu>=1.7.1'
!pip3 install 'numpy>=1.21.2'
!pip3 install 'Pillow>=9.0.1'
!pip3 install 'scikit_learn>=1.0.2'
!pip3 install 'tqdm>=4.62.3'
!pip3 install 'utm>=0.7.0'
!pip3 install 'kornia'
!pip3 install 'Shapely'
!pip3 install 'einops'
!pip3 install --upgrade --no-cache-dir gdown # support for download a large file from Google Drive
#use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #'cpu' # 'cuda' or 'cpu'
print(DEVICE)
```
## Downloading dataset - 224x224
These are the the datasets link in our drives
- https://drive.google.com/file/d/1Q_JGBHk5iN_lqA6OR5tBEZc_Lb-0wQ5b/view?usp=share_link sf_xs.zip
- https://drive.google.com/file/d/1Zya9NnGOZqAXo0b9Z4YfD5qPXpVq8p14/view?usp=share_link tokyo_xs.zip
- https://drive.google.com/file/d/1idC1UBdwSap_Nx1SZVDhRDHJI3LQn5E3/view?usp=share_link tokyo_night.zip
After setting up the libraries to donwload the dataset simply run:
```
from google.colab import drive
import os, sys

if not os.path.isfile('/content/sf_xs.zip'):
  !gdown 1Q_JGBHk5iN_lqA6OR5tBEZc_Lb-0wQ5b # 3-5 min (sf-xs)
  !jar xvf  "/content/sf_xs.zip"

if not os.path.isdir('/content/sf_xs'):
  print("Dataset doesn't exist")

if not os.path.isfile('/content/tokyo_xs.zip'):
  !gdown 1Zya9NnGOZqAXo0b9Z4YfD5qPXpVq8p14 # 3-5 min (tokyo-xs)
  !jar xvf  "/content/tokyo_xs.zip"            # estrae il file zip nella cartella (in questo caso small)

if not os.path.isdir('/content/tokyo_xs'):
  print("Dataset doesn't exist")

if not os.path.isfile('/content/tokyo_night.zip'):
  !gdown 1idC1UBdwSap_Nx1SZVDhRDHJI3LQn5E3 # 3-5 min (tokyo-night)
  !jar xvf  "/content/tokyo_night.zip"            # estrae il file zip nella cartella (in questo caso small)

if not os.path.isdir('/content/tokyo_night'):
  print("Dataset doesn't exist")
```
## Import code from github
We worked on 4 different branches
- `original`: for standard and loss functions training implementation
- `geowarp`: for geowarp training implementation
- `Delg`: for delg training implementation
- `backbone`: for cvt-cct training implementation

After cloning the repository import it by:
```
import sys
sys.path.append("/content/AG/")
import AG
from AG import *
```

## Loss functions
```
!git clone "https://github.com/GabriG23/AG"
!cd AG && git checkout main # add here the name of the branch 
```
After downloading the dataset, we can run
`!python3 AG/train.py --dataset_folder sf_xs --groups_num 1 --epochs_num 3 --loss_function loss_function_name`
With loss_functionname as cosface, arcface, sphereface.

#### Test
We can test a trained model as such:
`'python3 AG/eval.py --dataset_folder /content/sf_xs/ --backbone resnet18 --fc_output_dim 512 --resume_model path/to/best_model.pt`
`'python3 AG/eval.py --dataset_folder /content/tokyo_xs/ --backbone resnet18 --fc_output_dim 512 --resume_model path/to/best_model.pt`
`'python3 AG/eval.py --dataset_folder /content/tokyo_night/ --backbone resnet18 --fc_output_dim 512 --resume_model path/to/best_model.pt`
- In the folder `AG/trained_model` you can find all the different model with their hyperparameter
- Dataset_test_name as sf_xs, tokyo_xs or tokyo_night.
- We tested the model with resnet18 as backbone and output dimension of 512

## Geowarp
```
!git clone "https://github.com/GabriG23/AG"
!cd AG && git checkout geowarp
```
Training the feature extractor
`!python3 AG/train_fe.py --dataset_folder sf_xs --groups_num 1 --epochs_num 3 --num_workers 2`

Training the homography module
`!python3 AG/train_homography.py --dataset_folder sf_xs --groups_num 1 --epochs_num 3 --num_workers 2 --batch_size 32 --resume_fe /content/AG/trained_model/feature_extractor.pth`

- we can add a weight at the self-supervised loss with `--sw`, default = 1
- in `--resume_fe` we have to put the feature extractor path

#### Test
This is the comand to test the geowarp:
`!python3 AG/eval_geowarp.py --dataset_folder /content/sf_xs/ --num_reranked_predictions 20 --backbone resnet18 --fc_output_dim 512 --resume_fe /content/AG/trained_model/feature_extractor_resnet18.pth --resume_hr /content/AG/trained_model/homography_epoch3_batch32_ssw1.pth --num_workers 2`
- Modyfing `--dataset_folder` we can test on a different dataset
- we have to resume the feature extractor with `--resumer_fe` and the homography module with `--resume_hr`
## CVT-CCT

```
!git clone "https://github.com/GabriG23/AG"
!cd AG && git checkout backbone
import sys
sys.path.append("/content/AG/")
import AG
from AG import *
```
This is the command to train the module:
`!python3 AG/train_backbone.py --dataset_folder sf_xs --groups_num 1 --epochs_num 3 --num_workers 2 --batch_size 32 --backbone cvt --fc_output_dim 224`

#### Test
This is the comand to test the cvt:
`!python3 AG/eval.py --dataset_folder /content/sf_xs/ --backbone cvt --fc_output_dim 224 --resume_model /content/AG/trained_model/cvt_5epoch.pth`

## DELG

For the Delg implementation go to see 'Delg' branch.
