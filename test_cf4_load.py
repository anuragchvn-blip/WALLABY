import sys
sys.path.insert(0, 'e:/WALLABY')

from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4
import pandas as pd
import numpy as np

# Load the catalog
cat = read_cosmicflows4('e:/WALLABY/data/cosmicflows4.csv')

print(f"Loaded {cat.n_galaxies} galaxies")
print(f"RA range: {cat.ra.min():.2f} to {cat.ra.max():.2f}")
print(f"Dec range: {cat.dec.min():.2f} to {cat.dec.max():.2f}")

valid_cz = cat.cz[~np.isnan(cat.cz)]
print(f"Vcmb range: {valid_cz.min():.0f} to {valid_cz.max():.0f} km/s")

valid_dist = cat.distance[~np.isnan(cat.distance)]
print(f"Distance range: {valid_dist.min():.1f} to {valid_dist.max():.1f} Mpc")
print(f"Valid distances: {np.sum(~np.isnan(cat.distance))}")

# Method breakdown
methods, counts = np.unique(cat.method, return_counts=True)
print("\nMethods:")
for m, c in zip(methods, counts):
    print(f"  {m}: {c}")
