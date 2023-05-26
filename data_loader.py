from PIL import Image
import numpy as np

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
        img_tensor = torch.from_numpy(img_np).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor
        

    def create_mask(self, annotations):
        annotation_types = ['blood_vessel', 'glomerulus', 'unsure']
        # Initialize mask
        masks = np.zeros((3, 512, 512), dtype=np.float32)
        # Process annotations - aka fill in the mask
        for annot in annotations:
            assert len(annot['coordinates']) == 1  # This is the first assertion in my life that I've seen it's use
            cords = annot['coordinates'][0]        # I mean, I suppose it always has only one element, but to not having to check it and still not needing to worry....
            atype = annotation_types.index(annot['type'])
            # for cords in annot['coordinates']:   !this when the assert comes!
            for cord in cords:
                cord_np = np.asarray(cord, dtype=np.int32)
                cord_np = cord_np.T
                rr, cc = cord_np
                masks[atype][rr, cc] = 1
        # fill in the mask
        

        return masks
