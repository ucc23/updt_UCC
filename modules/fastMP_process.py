
from pathlib import Path
from . import call_fastMP
from . import main_process_GDR3_query as G3Q


def run(
    fastMP, new_DB, frames_path, frames_ranges, UCC_cat, GCs_cat, out_path
):
    """
    """

    # Create output folder in not present
    for quad in ('1', '2', '3', '4'):
        for s in ('P', 'N'):
            Qfold = 'Q' + quad + s
            Path(out_path + Qfold + '/datafiles/').mkdir(
                parents=True, exist_ok=True)

    call_fastMP.run(
        fastMP, G3Q, frames_path, frames_ranges, UCC_cat, GCs_cat, out_path,
        new_DB=new_DB)


if __name__ == '__main__':
    run()
