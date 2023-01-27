
import torch
from typing import Tuple, Union
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn



class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):  
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)

    # Viene ereditata da una trasformazione chiamata RandomResizedCrop in cui size è la dimensione dell'immagine di output
    # e scale è il range da cui verrà preso il valore random di zoom che sarà moltiplicato per l'immagine prima di
    # farne il resize
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:            # all'interno di qeusto forward c'è quello alla classe parent
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different ResizedCrop to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward      # prende la trasformazione della classe parent
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]    # la applica ad ogni immagine (è una lista)
        augmented_images = torch.cat(augmented_images)                                  # da lista a tensore in 4 dimensioni
        return augmented_images

    # Applicarla a tutte le immagini del dataset o durante il training piuttosto che farlo in questo modo cosa ha di diverso?
    # Ti evita di iterare sulle immagini durante il training in modo più esplicito?



if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])      
    # # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())                       # carica l'immagine di un astronauta da un database
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)       # traforma in tensore aggiunge una dimensione come asse 0 (quindi contenitore esterno)
    images_batch = torch.cat([tensor_image, tensor_image])              # concatena i tensori alla dimensione 0 (di default). Motvi per il quale unsqueeze(0) 
                                                                        # è necessario sennò somma sui canali e si avrebbe (6, 512, 512)
                                                                        # images_batch ha quindi shape (2, 3, 512, 512) -> le due immagini, canali, matrice
    # Apply augmentation (individually on each of the 2 images)
    # augmented_batch = random_crop(images_batch)                         # applica il random crop di classe DeviceAgnosticRandomResizedCrop al batch (chiama forward)
    # Convert to PIL images   
    # augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])   # riconverte le immagini indietro così da poterle mostrare
    # augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    # pil_image.show()
    # augmented_image_0.show()
    # augmented_image_1.show()

    # images_batch = torch.cat([tensor_image]) 
    # print(images_batch)
    # depth_wise_max = torch.max(images_batch, dim=1)[0]
    # print(depth_wise_max)
    # is_depth_wise_max = (images_batch == depth_wise_max)
    # print(is_depth_wise_max)

    print(images_batch.shape)                   # -> (2, 3, 512, 512)
    print(images_batch[:, :, 0, 0].shape)       # -> (2, 3)

    # all_keypoints = torch.zeros([3, 0]) 
    # print(all_keypoints.shape)

    # local_max = F.max_pool2d(images_batch, 3, stride=1, padding=1)
    # print(local_max.shape)
    # b, c, h, w = images_batch.size()
    # print(images_batch.view(-1, 1, h, w).shape)
