import os
from pathlib import Path

from fast_diff_match_patch import diff


def update_image(
    DRY_RUN: bool,
    logging,
    plot_path_old: str,
    plot_pars: tuple,
) -> str:
    """
    Updates or generates an image file based on the specified parameters.

    Args:
        DRY_RUN (bool): If True, no changes are made to the file system.
        logging (object): Logger instance used for logging file operations.
        plot_path_old (str): Path to the existing image file.
        plot_pars (tuple): Parameters for generating the plot. This includes:
            - The actual function to generate the plot as the first element
            - Extra parameters required which depend on the plot itself

    Returns:
        str: Status of the operation, which can be one of the following:
            - 'generated': A new image file was created.
            - 'updated': An existing image file was replaced with an updated version.
            - '': No changes were made (e.g., the new image matches the old one).
    """
    # Unpack plotting function
    selected_plot = plot_pars[0]

    if Path(plot_path_old).is_file() is False:
        # If 'old' image files does not exist --> generate
        if DRY_RUN is False:
            selected_plot(plot_path_old, *plot_pars[1:])
        txt0 = "generated"
    else:
        # Generate name of new (temporary) file
        ext = "." + plot_path_old.split(".")[-1]
        plot_path_new = plot_path_old.replace(ext, "_new" + ext)
        selected_plot(plot_path_new, *plot_pars[1:])

        # If the new image is different to the old one --> delete old + rename new
        if not are_images_equal(plot_path_old, plot_path_new):
            if DRY_RUN is False:
                # Delete old image
                delete_file(plot_path_old, logging)
                # Rename new image to old name
                rename_file(plot_path_new, plot_path_old, logging)
            else:
                delete_file(plot_path_new, logging)
            txt0 = "updated"
        else:
            # New image is equal to the old one --> delete new file
            delete_file(plot_path_new, logging)
            txt0 = ""

    return txt0


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

        # Fast check
        if len(image1_data) != len(image2_data):
            return False

        # Use this library to compare binary data of equal length
        changes = diff(image1_data, image2_data)
        var = changes[0][0] == "=" and changes[0][1] == len(image1_data)
        return var

        # # Use difflib to compare binary data
        # diff = difflib.SequenceMatcher(None, image1_data, image2_data)
        # return diff.ratio() == 1.0

        # # This library is much faster
        # dmp = diff_match_patch()
        # patches = dmp.patch_make(str(image1_data), str(image2_data))
        # var = len(patches) == 0
        # return var

    except Exception as e:
        print(f"Error: {e}")
        return False


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
