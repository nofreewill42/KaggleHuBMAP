from pathlib import Path



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

    # Read in the csv files
    wsi_meta = pd.read_csv(ds_path / 'wsi_meta.csv')
    tile_meta = pd.read_csv(ds_path / 'tile_meta.csv')
    
    