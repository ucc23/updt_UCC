from os.path import join
from pathlib import Path
import datetime
import logging


def main():
    """ """
    mypath = Path().absolute()
    # Go up one level
    mypath = "/".join(str(mypath).split("/")[:-1])

    # Name of log file using the date
    x = datetime.date.today()
    out_file = str(x).replace("-", "_") + ".log"

    # Set up logging module
    level = logging.INFO
    frmt = "%(message)s"
    handlers = [
        logging.FileHandler(join(mypath, out_file), mode="a"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=level, format=frmt, handlers=handlers)

    logging.info("\n------------------------------")
    logging.info(str(datetime.datetime.now()) + "\n")

    return logging


if __name__ == "__main__":
    main()
