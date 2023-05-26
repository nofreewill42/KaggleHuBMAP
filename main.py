import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from torch.utils.data import Dataset



if __name__ == '__main__':
    # Read in ds_path.txt content and make a pathlib.Path object
    ds_path = Path(Path('ds_path.txt').read_text().strip())

    '''
    nofreewill@nofreewill:/media/nofreewill/8TB-SSD/Visual/hubmap-hacking-the-human-vasculature$ ls
hubmap-hacking-the-human-vasculature.zip  test           wsi_meta.csv
polygons.jsonl                            tile_meta.csv
sample_submission.csv                     train
nofreewill@nofreewill:/media/nofreewill/8TB-SSD/Visual/hubmap-hacking-the-human-vasculature$
    '''
    
    '''
    Files and Field Descriptions

    {train|test}/ Folders containing TIFF images of the tiles. Each tile is 512x512 in size.
    polygons.jsonl Polygonal segmentation masks in JSONL format, available for Dataset 1 and Dataset 2. Each line gives JSON annotations for a single image with:
        id Identifies the corresponding image in train/
        annotations A list of mask annotations with:
        type Identifies the type of structure annotated:
            blood_vessel The target structure. Your goal in this competition is to predict these kinds of masks on the test set.
            glomerulus A capillary ball structure in the kidney. These parts of the images were excluded from blood vessel annotation. You should ensure none of your test set predictions occur within glomerulus structures as they will be counted as false positives. Annotations are provided for test set tiles.
            unsure A structure the expert annotators cannot confidently distinguish as a blood vessel.
        coordinates A list of polygon coordinates defining the segmentation mask.
    tile_meta.csv Metadata for each image.
        source_wsi Identifies the WSI this tile was extracted from.
        {i|j} The location of the upper-left corner within the WSI where the tile was extracted.
        dataset The dataset this tile belongs to, as described above.
    wsi_meta.csv Metadata for the Whole Slide Images the tiles were extracted from.
        source_wsi Identifies the WSI.
        age, sex, race, height, weight, and bmi demographic information about the tissue donor.'''
    
    # Read in the tile_meta.csv file
    tile_meta = pd.read_csv(ds_path / 'tile_meta.csv')
    
    # Read in the wsi_meta.csv file
    wsi_meta = pd.read_csv(ds_path / 'wsi_meta.csv')

    # Read in the polygons.jsonl file
    polygons = pd.read_json(ds_path / 'polygons.jsonl', lines=True)

    # Merge the DataFrames
    tile_wsi = tile_meta.merge(wsi_meta, on='source_wsi', how='left')
    tile_polygons = tile_wsi.merge(polygons, on='id', how='left')

    # pytorch dataset class
    class HuBMAPDataset(Dataset):
        def __init__(self, tile_polygons):
            # Drop everything except dataset type 1
            self.tile_polygons = tile_polygons[tile_polygons['dataset'] == 1]

        def __len__(self):
            return len(self.tile_polygons)
        
        def __getitem__(self, idx):
            # Read in the image
            img_pil = Image.open(ds_path / 'train' / (self.tile_polygons.iloc[idx]['id'] + '.tif'))
            annotations = self.tile_polygons.iloc[idx]['annotations']

            img_np = np.asarray(img_pil, dtype=np.float32)/255
            mask_np = self.create_mask(annotations)

            return img_np, mask_np
            

        def create_mask(self, annotations):
            # Initialize mask
            mask = np.zeros((512, 512), dtype=np.float32)
            # Process annotations - aka fill in the mask
            for annot in annotations:
                assert len(annot['coordinates']) == 1  # This is the first assertion in my life that I've seen it's use
                cords = annot['coordinates'][0]        # I mean, I suppose it always has only one element, but to not having to check it and still not needing to worry....
                if annot['type'] == "blood_vessel":
                    for cord in cords:
                        cord_np = np.asarray(cord, dtype=np.int32)
                        cord_np = cord_np.T
                        rr, cc = cord_np
                        mask[rr, cc] = 1
            return mask

    ds = HuBMAPDataset(tile_polygons)

    print(ds[0])