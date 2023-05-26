from PIL import Image
import numpy as np
import cv2


import torch
from torch.utils.data import Dataset


class HuBMAPDataset(Dataset):
    def __init__(self, ds_path, tile_polygons):
        self.ds_path = ds_path
        self.tile_polygons = tile_polygons[tile_polygons['dataset'] == 1] # Drop everything except dataset type 1

    def __len__(self):
        return len(self.tile_polygons)
    
    def __getitem__(self, idx):
        # Read in the image
        img_pil = Image.open(self.ds_path / 'train' / (self.tile_polygons.iloc[idx]['id'] + '.tif'))
        annotations = self.tile_polygons.iloc[idx]['annotations']

        img_np = np.asarray(img_pil, dtype=np.float32)/255
        mask_np = self.create_mask(annotations)

        # convert to torch tensor
        img_tensor = torch.from_numpy(img_np).permute(2,0,1).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor
        

    def create_mask(self, annotations):
        annotation_types = ['blood_vessel', 'glomerulus', 'unsure']
        # Initialize mask
        masks = np.zeros((3, 512, 512), dtype=np.float32)
        # Process annotations - aka fill in the mask
        for annot in annotations:
            atype = annotation_types.index(annot['type'])
            cords = annot['coordinates']
            for cord in cords:
                lines = np.array(cord)
                lines = lines.reshape(-1, 1, 2)
                cv2.fillPoly(masks[atype], [lines], 255)
        return masks
