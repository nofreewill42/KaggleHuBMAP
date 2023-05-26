import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader

from data_loader import HuBMAPDataset



if __name__ == '__main__':
    # Read in ds_path.txt content and make a pathlib.Path object
    ds_path = Path(Path('ds_path.txt').read_text().strip())
    
    # Read in the tile_meta.csv file
    tile_meta = pd.read_csv(ds_path / 'tile_meta.csv')
    
    # Read in the wsi_meta.csv file
    wsi_meta = pd.read_csv(ds_path / 'wsi_meta.csv')

    # Read in the polygons.jsonl file
    polygons = pd.read_json(ds_path / 'polygons.jsonl', lines=True)

    # Merge the DataFrames
    tile_wsi = tile_meta.merge(wsi_meta, on='source_wsi', how='left')
    tile_polygons = tile_wsi.merge(polygons, on='id', how='left')


    ds = HuBMAPDataset(ds_path, tile_polygons)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, prefetch_factor=2)

    for i, (img, mask) in enumerate(dl):
is         print(i, img.shape, mask.shape)