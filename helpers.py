import os


def get_datasets_paths(config, type):
    paths = []
    for dataset_name in config['datasets']:
        paths.append(os.path.join(config['datasets_dir'], dataset_name, type, type + '.csv'))
    return paths