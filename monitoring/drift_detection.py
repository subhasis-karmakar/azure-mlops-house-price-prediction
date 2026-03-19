from scipy.stats import ks_2samp
def detect_drift(ref_df, new_df):
    drift = {}
    for col in ref_df.columns:
        _, p = ks_2samp(ref_df[col], new_df[col])
        drift[col] = p
    return drift
