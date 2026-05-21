"""Automated downloader for WorldClim 2.1 Bioclimatic variables.

Downloads the 2.5m resolution ZIP, extracts the 19 GeoTIFFs, 
and organizes them for the Ecological DB Builder.
"""

import os
import zipfile
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """
    Tqdm extension for progress bars with file download support.

    Parameters
    ----------
    None
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """
        Progress hook compatible with `urllib.request.urlretrieve`.

        Parameters
        ----------
        b : int, optional
            Number of blocks transferred so far, by default 1.
        bsize : int, optional
            Size of each block (in tqdm units), by default 1.
        tsize : int, optional
            Total size (in tqdm units), by default None.

        Returns
        -------
        None
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    """
    Download a file from a URL with a progress bar.

    Parameters
    ----------
    url : str
        The remote URL to download from.
    output_path : str
        The local path where the file will be saved.

    Returns
    -------
    None
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main() -> None:
    """
    Main function to download and extract WorldClim 2.1 Bioclimatic variables.

    Downloads the ZIP from a mirror, extracts the GeoTIFFs, and organizes them.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Use the working geodata mirror instead of biogeo
    URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"
    DEST_DIR = "/workspace/plantclef/raw/worldclim"
    ZIP_PATH = os.path.join(DEST_DIR, "wc2.1_2.5m_bio.zip")
    
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # 1. Download
    if not os.path.exists(ZIP_PATH):
        print(f"[WorldClim] Downloading from {URL}...")
        download_url(URL, ZIP_PATH)
    else:
        print(f"[WorldClim] Zip already exists at {ZIP_PATH}")
        
    # 2. Extract
    print(f"[WorldClim] Extracting to {DEST_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
        
    # 3. Clean up
    os.remove(ZIP_PATH)
    print(f"[WorldClim] Success! Files ready in {DEST_DIR}")

if __name__ == "__main__":
    main()
