"""Download the PeMS04 traffic-flow dataset (optional) and print its structure.

Source: a Hugging Face mirror of the public PeMS04 benchmark (San Francisco Bay Area,
307 sensors, 5-minute readings, Jan-Feb 2018). The original data comes from Caltrans PeMS
and is commonly obtained via the ASTGCN project. Check the dataset's terms before reuse.

If this download fails, get the file manually in a browser from
https://huggingface.co/datasets/bjdwh/FlashST-DATA/tree/main/PEMS04
and save it as data/PEMS04.npz.
"""
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/bjdwh/FlashST-DATA/resolve/main/PEMS04/PEMS04.npz"
out = Path("data")
out.mkdir(exist_ok=True)
dest = out / "PEMS04.npz"

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
print("Downloading (about 33 MB)...")
with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
    f.write(response.read())
size_mb = dest.stat().st_size / 1e6
print(f"Saved to {dest} ({size_mb:.1f} MB)")
if size_mb < 1:
    raise SystemExit("File is too small - probably an error page, not the dataset. Download it manually.")

# Safe loading: allow_pickle stays False, so a malicious pickle cannot run code.
with np.load(dest) as z:
    for k in z.files:
        print(f"key={k!r} shape={z[k].shape}")
print("Expected: key 'data' with shape (16992, 307, 3). Feature 0 (flow) is used by main.py.")
