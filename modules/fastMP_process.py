
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

    if new_DB is not None:
        # Only process 'new_DB' (if given)
        msk_new_clusters = []
        for index, cl in df_UCC.iterrows():
            if new_DB in cl['DB']:
                msk_new_clusters.append(True)
            else:
                msk_new_clusters.append(False)
        clusters_list = df_UCC[np.array(msk_new_clusters)]
    else:
        # Full list
        clusters_list = df_UCC

    call_fastMP.run(
        fastMP, G3Q, frames_path, frames_data, df_UCC, df_gcs, UCC_cat,
        out_path, clusters_list)


if __name__ == '__main__':
    run()
