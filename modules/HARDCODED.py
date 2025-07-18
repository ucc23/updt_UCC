# UCC site paths
#
# Root path to the repo for the ucc site
ucc_path = "ucc/"
# Path to the md entries folder
md_folder = ucc_path + "_clusters/"
# Path to the pages folder
pages_folder = ucc_path + "_pages/"
# Path to the tables folders
tables_folder = ucc_path + "_tables/"
dbs_tables_folder = tables_folder + "dbs/"
# Path to the ucc site images folder
images_folder = ucc_path + "images/"
# Assets folder
assets_folder = ucc_path + "assets/"
# Path to the compressed CSV file with clusters data
clusters_csv_path = assets_folder + "clusters.csv.gz"
# Path to the md database file
databases_md_path = pages_folder + "DATABASE.md"
# # Path to the md file with the tables
# tables_md_path = pages_folder + "TABLES.md"

# Q folders paths and files
#
# Path to the .parquet member files
members_folder = "datafiles/"
# Path to the cluster's plots
plots_folder = "plots/"


# updt_ucc paths
#
# Path to the DBs and their JSON files are stored
dbs_folder = "databases/"
# Path to the JSON file with DBs data
name_DBs_json = dbs_folder + "all_dbs.json"
# Path to the database of GCs
GCs_cat = dbs_folder + "globulars.csv"
# Path to the file with manual OC parameters
manual_pars_file = "manual_params.csv"
# Path to store temporarily all updated and new files
temp_fold = "temp_updt/"

# Zenodo paths and names
#
# Path to the folder that stores the latest UCC version
UCC_folder = "zenodo/"
# Path to the folder that stores the archived UCC versions
UCC_archive = UCC_folder + "archived/"
# Name of the  file that contains all of the members
UCC_members_file = "UCC_members.parquet"
# File that stores the dates and status of each member extraction
parquet_dates = "data_dates.json"


# Paths to the Gaia DR3 files in external drive
#
# Path to the folder that contains the raw data
root = "/media/gabriel/backup/gabriel/GaiaDR3/"
# root = "/home/gperren.ifir/UCC/GaiaDR3/"
path_gaia_frames = root + "datafiles_G20/"
# Path to the file that informs the sky area covered by each raw data file
path_gaia_frames_ranges = root + "files_G20/frame_ranges.txt"
# Maximum magnitude to retrieve
gaia_max_mag = 20


# Path to local ASteCA version
local_asteca_path = "/home/gabriel/Github/ASteCA/ASteCA/asteca"
