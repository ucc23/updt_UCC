
import configparser


def main():
    """
    Read .ini config file
    """

    in_params = configparser.ConfigParser()
    in_params.read('../params.ini')
    pars_dict = {}

    pars = in_params['Re-process OCs']
    pars_dict['OCs_names'] = pars.get('OCs_names')

    pars = in_params['General']
    pars_dict['dbs_folder'] = pars.get('dbs_folder')
    pars_dict['new_DB'] = pars.get('new_DB')
    pars_dict['GCs_cat'] = pars.get('GCs_cat')
    pars_dict['UCC_folder'] = pars.get('UCC_folder')
    pars_dict['sep'] = pars.get('sep')

    pars = in_params['New DB check']
    pars_dict['search_rad'] = pars.getfloat('search_rad')
    pars_dict['cID'] = pars.get('cID')
    pars_dict['clon'] = pars.get('clon')
    pars_dict['clat'] = pars.get('clat')
    pars_dict['coords'] = pars.get('coords')
    pars_dict['rad_dup'] = pars.getfloat('rad_dup')

    pars = in_params['Check/Add new DB']
    pars_dict['all_DBs_json'] = pars.get('all_DBs_json')
    pars_dict['new_OCs_fpath'] = pars.get('new_OCs_fpath')

    pars = in_params['Run fastMP / Updt UCC']
    pars_dict['frames_path'] = pars.get('frames_path')
    pars_dict['frames_ranges'] = pars.get('frames_ranges')
    pars_dict['max_mag'] = pars.getfloat('max_mag')
    pars_dict['out_path_membs'] = pars.get('out_path_membs')
    pars_dict['manual_pars_f'] = pars.get('manual_pars_f')
    pars_dict['verbose'] = pars.getint('verbose')
    pars_dict['new_OCs_data'] = pars.get('new_OCs_data')

    pars = in_params['Check versions']
    pars_dict['old_UCC_name'] = pars.get('old_UCC_name')

    pars = in_params['Make entries']
    pars_dict['root_UCC_path'] = pars.get('root_UCC_path')
    pars_dict['md_folder'] = pars.get('md_folder')
    pars_dict['pages_folder'] = pars.get('pages_folder')
    pars_dict['clusters_json'] = pars.get('clusters_json')
    pars_dict['members_folder'] = pars.get('members_folder')
    pars_dict['ntbks_folder'] = pars.get('ntbks_folder')
    pars_dict['plots_folder'] = pars.get('plots_folder')

    return pars_dict
