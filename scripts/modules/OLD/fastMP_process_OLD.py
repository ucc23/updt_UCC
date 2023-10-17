
import numpy as np
from . import call_fastMP
from . import main_process_GDR3_query as G3Q


def run(
    fastMP, new_DB, frames_path, frames_ranges, UCC_cat, GCs_cat, out_path
):
    """
    """
    # Read data
    frames_data, df_UCC, df_gcs = call_fastMP.read_input(
        frames_ranges, UCC_cat, GCs_cat)

    if new_DB is None:
        # Full list
        clusters_list = df_UCC
    else:
        # Only process 'new_DB' (if given)
        msk_new_clusters = []
        for index, cl in df_UCC.iterrows():
            if new_DB in cl['DB']:
                msk_new_clusters.append(True)
            else:
                msk_new_clusters.append(False)
        clusters_list = df_UCC[np.array(msk_new_clusters)]

    call_fastMP.run(
        fastMP, G3Q, frames_path, frames_data, df_UCC, df_gcs, UCC_cat,
        out_path, clusters_list)

    # This dataframe returns changed by 'call_fastMP', i.e.: it is not in the
    # same state as the version of the dataframe loaded at the top of this
    # script (which is what we want)
    return df_UCC


if __name__ == '__main__':
    run()
