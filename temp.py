import pandas as pd 
from itertools import chain

# train = pd.read_csv(r'C:/Users/User/dev/data/q1_q17_dr24_tce_clean.csv')
# test = pd.read_csv(r'C:/Users/User/dev/data/q1_q17_dr24_tce_clean_test.csv')


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


df = pd.read_csv(r'C:/Users/User/dev/data/q1_q17_dr24_tce_full.csv', comment='#')


print(set(columns).issubset(df.columns))

# with open('./data/kepids.txt', 'w') as f:
#     for id_ in set(df['kepid']):
#         f.write(f'{id_}\n')

# print(len(train.columns), train.columns)
# print(len(df.columns), df.columns)



