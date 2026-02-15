"""Debug script to trace the pipeline issue."""
import sys
sys.path.insert(0, 'e:/WALLABY')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4
from velocity_reconstruction.preprocessing.malmquist import apply_malmquist_correction
from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity

cat = read_cosmicflows4('e:/WALLABY/data/cosmicflows4.csv')
print('1. Initial galaxies:', len(cat.cz))

# Malmquist
distance_corr = cat.distance.copy()
for method in np.unique(cat.method):
    if method == 'Unknown':
        continue
    mask = cat.method == method
    mag_lim = {'TF': 14.5, 'FP': 16.0, 'SN Ia': 17.5}.get(method, 15.0)
    distance_corr[mask] = apply_malmquist_correction(
        cat.distance[mask], cat.distance_error[mask],
        method=method, magnitude_limit=mag_lim
    )
print('2. After Malmquist, valid distances:', np.isfinite(distance_corr).sum())

# Valid mask
valid_dist_mask = (distance_corr > 0.5) & (distance_corr < 1000) & np.isfinite(distance_corr)
print('3. Valid distance mask:', valid_dist_mask.sum())

# v_pec
v_pec = np.full_like(cat.cz, np.nan)
v_pec[valid_dist_mask] = compute_peculiar_velocity(cat.cz[valid_dist_mask], distance_corr[valid_dist_mask], H0=70.0)
print('4. v_pec computed:', np.isfinite(v_pec).sum())

# After clip
v_pec_clipped = np.clip(v_pec, -3000, 3500)
print('5. After clip:', np.isfinite(v_pec_clipped).sum())

# Check what the pipeline says
print('\n=== Pipeline says: ===')
print('Valid distances for v_pec: 572/9999')
print('This suggests something else is filtering.')
