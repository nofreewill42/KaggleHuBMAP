from pathlib import Path



if __name__ == '__main__':
# Read in ds_path.txt content and make a pathlib.Path object
ds_path = Path(Path('ds_path.txt').read_text().strip())