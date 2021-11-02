import pandas as pd
import numpy as np
import os 
from .helpers import download_target_lightcurve, norm_features, norm_kepid
from .preprocessing import * 

from astropy.io import fits
from functools import lru_cache


columns = ['tce_period',
           'tce_time0bk',
           'tce_impact',
           'tce_duration',
           'tce_depth',
           'tce_model_snr',
           'tce_prad',
           'tce_eqt',
           'tce_steff',
           'tce_slogg',
           'tce_sradius',
           'boot_fap',
           'boot_mesmean',
           'boot_messtd',
           'boot_mesthresh',
           'tce_chisq1',
           'tce_chisq2',
           'tce_dicco_mdec',
           'tce_dicco_mra',
           'tce_dicco_msky',
           'tce_dikco_mdec',
           'tce_dikco_mra',
           'tce_dikco_msky',
           'tce_dof1',
           'tce_dof2',
           'tce_dor',
           'tce_fwm_pdeco',
           'tce_fwm_prao',
           'tce_fwm_sdec',
           'tce_fwm_sdeco',
           'tce_fwm_sra',
           'tce_fwm_srao',
           'tce_fwm_stat',
           'tce_ldm_coeff1',
           'tce_ldm_coeff2',
           'tce_ldm_coeff3',
           'tce_ldm_coeff4',
           'tce_max_mult_ev',
           'tce_max_sngle_ev',
           'tce_model_chisq',
           'tce_model_dof',
           'tce_num_transits',
           'tce_robstat',
           'tce_ror',
           ]

normed_columns = [col+"_norm" for col in columns]
train_root_dir = "C:/Users/User/dev/data/train/"


def to_int_label(label, binary):
    if not binary:
        if label == 'PC':
            return 2
        if label == 'AFP':
            return 1
        if label == 'NTP':
            return 0
        raise ValueError('wrong', label)
    else:
        if label == 'PC':
            return 1
        return 0

def sort_df(df, binary=True):
    df['norm_kepid'] = df['kepid'].apply(norm_kepid)
    df['int_label'] = df['av_training_set'].apply(lambda x: to_int_label(x, binary=binary))
    df.sort_values(by=['int_label', 'norm_kepid', 'tce_plnt_num'],
                   ascending=[False, True, True],
                   inplace=True, kind='mergesort')

    df = df.reset_index(drop=True)
    return df


def get_time_flux_by_ID(kepid,
                        quarter=None,
                        func_kepid_to_lc_files=None,
                        inject_no=3,
                        scramble_id=None):

    def default_func_kepid_to_lc_files(kepid, inject_no=None, root_dir=None):
        root_dir = os.path.join(train_root_dir, kepid[:4], kepid)
        files = sorted(os.listdir(root_dir))
        return [os.path.join(root_dir, f) for f in files]

    if func_kepid_to_lc_files is None:
        func_kepid_to_lc_files = default_func_kepid_to_lc_files

    all_time, all_flux = [], []
    kepid = f'{int(kepid):09d}'
    prev_quarter = -1
    try:
        # root_dir = os.path.join(train_root_dir, kepid[:4], kepid)
        # files = os.listdir(root_dir)
        # files = sorted(files)
        files = func_kepid_to_lc_files(kepid, inject_no=inject_no, root_dir=None)
        for file in files:
            # file = os.path.join(root_dir, file)
            with fits.open(file) as hdu:
                cur_quarter = int(hdu[0].header['quarter'])
                if cur_quarter <= prev_quarter:
                    raise ValueError("files are not sorted")
                prev_quarter = cur_quarter
                flux = hdu[1].data.PDCSAP_FLUX
                time = hdu[1].data.TIME
                # remove nan
                finite_mask = np.isfinite(flux)
                time = time[finite_mask]
                flux = flux[finite_mask]
                flux /= np.median(flux)

                if quarter is not None and str(cur_quarter) == str(quarter):
                    return time, flux

                ##################################
                # later add remove outliers ???  #
                #                                #
                ##################################

                all_time.append(time)
                all_flux.append(flux)
    except Exception as e:
        print(e)
        # download the file if failes opening
        max_retries = 10
        for i in range(max_retries):
            if download_target_lightcurve(kepid):
                return get_time_flux_by_ID(kepid, quarter=quarter, scramble_id=scramble_id)
        else:
            return None, None

    # quarter is beyond [1,17]
    if quarter is not None:
        print(f'quarter {quarter} not exist')
        return None

    all_time = np.concatenate(all_time)
    all_flux = np.concatenate(all_flux)

    return all_time, all_flux



def load_model(path=None):
    import tensorflow as tf 
    path = os.path.join(os.path.dirname(__file__),
                        "train.h5") if path is None else path

    try:
        print(f'loading model from {path}')
        model = tf.keras.models.load_model(path)
        print(f'finished loading')
        return model
    except Exception as e:
        print(f'Error loading model from {path}')
        print("loading error: ", e)
        os._exit(-1)


@lru_cache(maxsize=256)
def read_csv(csv_name):
    df_clean = pd.read_csv(csv_name, comment='#')
    df_clean = sort_df(df_clean)
    df_clean['tce_duration'] = df_clean['tce_duration']/24.0
    return df_clean


def test_kepid(models, kepid, params=None, verbose=False, csv_name=None):
    """
    if params is not None, duration is in Hours
    """

    if not isinstance(models, list):
        models = [models]

    df_clean = read_csv(csv_name)
    targets = df_clean[df_clean['kepid'] == int(kepid)]

    # try:
    #     test_features = np.round(targets[normed_columns].values, 6)
    # except KeyError:
        # write norm_columns
    df_clean['duration/period'] = df_clean['tce_duration']/df_clean['tce_period']
    for col, norm_col in zip(columns, normed_columns):
        vals = df_clean[col].values.reshape(-1, 1)
        normed_vals = norm_features(vals)
        df_clean[norm_col] = normed_vals
    df_clean = sort_df(df_clean)
    targets = df_clean[df_clean['kepid'] == int(kepid)]
    test_features = np.round(targets[normed_columns + ['duration/period']].values, 6)

    if len(targets) == 0:
        return {}
    time, flux = get_time_flux_by_ID(kepid)

    planet_nums, period_list, t0_list, duration_list = targets[[
        'tce_plnt_num', 'tce_period', 'tce_time0bk', 'tce_duration']].values.T

    summary = {}
    total = len(period_list)

    remove_indices = list(range(total))
    remove_periods = [period_list[i] for i in remove_indices]
    remove_t0s = [t0_list[i] for i in remove_indices]
    remove_durations = [duration_list[i] for i in remove_indices]

    for i, (period, t0, duration, planet_num) in \
            enumerate(zip(
                period_list,
                t0_list,
                duration_list,
                planet_nums
            )):
        if verbose:
            write_info(f'loading {i + 1}/{total}')

        cur_time, cur_flux = remove_points_other_tce(
            time, flux, period, period_list=remove_periods,
            t0_list=remove_t0s, duration_list=remove_durations
        )

        global_flux = process_global(
            cur_time, cur_flux, period, t0, duration
        )

        local_flux = process_local(
            cur_time, cur_flux, period, t0, duration, center_ratio=7
        )
        # reshape flux
        global_flux = global_flux.reshape(1, *global_flux.shape, 1)
        local_flux = local_flux.reshape(1, *local_flux.shape, 1)

        global_flux = np.round(global_flux, 3)
        local_flux = np.round(local_flux, 3)

#         print(global_flux[0][:5])
#         print(local_flux[0][:5])
        predictions = 0
        test_feature = test_features[i]

#         print('--->', test_feature)
        for j, model in enumerate(models):
            # print('predicting', i)
            pred = model.predict([global_flux,
                                  local_flux,
                                  test_feature.reshape(1, *test_feature.shape)])
#             print(len(pred))
            predictions += pred[0, 1]
        predictions /= len(models)
        summary[int(planet_num)] = predictions
    return summary

