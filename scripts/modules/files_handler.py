import difflib
import os
from pathlib import Path

import matplotlib.pyplot as plt


def rename_file(old_name, new_name, logging):
    """
    Rename a file using the os library.
    :param old_name: Current name of the file (including the path if not in the same
                     directory)
    :param new_name: New name for the file (including the path if moving to another
                     directory)
    """
    try:
        os.rename(old_name, new_name)
    except FileNotFoundError:
        logging.info(f"Error: The file '{old_name}' does not exist.")
    except PermissionError:
        logging.info("Error: Permission denied.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")


def delete_file(file_path, logging):
    """
    Delete a file using the os library.
    :param file_path: Path to the file to be deleted
    """
    try:
        os.remove(file_path)
    except FileNotFoundError:
        logging.info(f"Error: The file '{file_path}' does not exist.")
    except PermissionError:
        logging.info("Error: Permission denied.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")


def update_image(DRY_RUN, logging, fig, path_old, dpi):
    """ """
    # Generate name of new (temporary) file
    ext = "." + path_old.split(".")[-1]
    path_new = path_old.replace(ext, "_new" + ext)

    # Generate new image
    plt.savefig(path_new, dpi=dpi)

    # If 'old' image files does not exist --> rename new as old (generate)
    if Path(path_old).is_file() is False:
        if DRY_RUN is False:
            # Rename new image to old name
            rename_file(path_new, path_old, logging)
        else:
            delete_file(path_new, logging)
        txt = "generated"
    else:
        # Check if the new image and the old one are identical
        flag = are_images_equal(path_old, path_new)

        # If the new image is different to the old one
        if flag is False:
            if DRY_RUN is False:
                # Delete old image
                delete_file(path_old, logging)
                # Rename new image to old name
                rename_file(path_new, path_old, logging)
            else:
                delete_file(path_new, logging)
            txt = "updated"
        else:
            # New image is equal to the old one --> delete new file
            delete_file(path_new, logging)
            txt = ""

    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)

    return txt


def are_images_equal(image1_path, image2_path):
    """
    Compare two image files using difflib at the binary level.
    :param image1_path: Path to the first image file
    :param image2_path: Path to the second image file
    :return: True if images are equal, False otherwise
    """
    try:
        # Read the files in binary mode
        with open(image1_path, "rb") as file1, open(image2_path, "rb") as file2:
            image1_data = file1.read()
            image2_data = file2.read()

        # Use difflib to compare binary data
        diff = difflib.SequenceMatcher(None, image1_data, image2_data)
        return diff.ratio() == 1.0
    except Exception as e:
        print(f"Error: {e}")
        return False
