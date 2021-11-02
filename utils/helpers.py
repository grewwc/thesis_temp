try:
    from lightkurve import search_lightcurve
except:
    search_lightcurve = None


from config import kepler_data_rootdir


from tempfile import TemporaryDirectory
import os
import shutil

import numpy as np
from functools import lru_cache
import pandas as pd


def norm_kepid(kepid):
    return f'{int(kepid):09d}'


def download_target_lightcurve(target, mission='kepler'):
    prefix = 'KIC'
    if mission.lower() == 'tess':
        prefix = 'TIC'
    print(f'prepare to download {prefix} {target} to {kepler_data_rootdir}')
    lc = search_lightcurve(f'{prefix} {target}')

    if len(lc) == 0:
        print(f'can not find {target}')
        return False

    with TemporaryDirectory() as tempdir:
        lc = lc.download_all(download_dir=tempdir)
        if lc is None:
            print(f'downloading {prefix} {target} failes')
            return False
        kepler_prefix = os.path.join(tempdir, 'mastDownload', 'Kepler')
        subdirs = os.listdir(kepler_prefix)

        if len(subdirs) != 1:
            raise ValueError("should only have one subdir")
        kepler_prefix = os.path.join(kepler_prefix, subdirs[0])
        target = f'{int(target):09d}'
        outdir = os.path.join(kepler_data_rootdir, target[:4], target)
        for lc in os.listdir(kepler_prefix):
            shutil.move(os.path.join(kepler_prefix, lc), os.path.join(outdir, lc))

    print(f'finished downloading {target}')

    return True


def norm_features(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    try:
        values = (values - mean) / std
    except Warning as e:
        print(e, std)
    return values


@lru_cache(maxsize=100)
def get_robo_pred(csv_name=None):
    if csv_name is None:
        csv_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'robo-24.csv')
    df = pd.read_csv(csv_name)
    df['norm_kepid'] = df['kepid'].apply(norm_kepid)
    return df


download_target_lightcurve('307210830', mission='tess')
