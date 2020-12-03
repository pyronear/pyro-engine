import configparser


def read_config_file(filename):
    """Read the config file and return it as a dictionary."""
    config = configparser.ConfigParser()
    config.read(filename)
    conf = {}
    for parts in config:
        for key in config[parts]:
            conf[key] = config[parts][key]

    return conf


def write_config_file(filename='inference.cfg'):
    """Use config parser to save config file."""
    config = configparser.ConfigParser()
    config['MODEL'] = {'backbone': 'rexnet1_0x',
                       'num_classes': 1,
                       'use_adaptive_concat_pool2d_head': True,
                       'nb_features': 1280,
                       'cut': -2,
                       'checkpoint': 'pyro_checkpoint_V0.1.pth',
                       'device': 'cuda'}

    config['TRANSFORMS'] = {'image_size': 448,
                            'use_center_crop': True}

    with open(filename, 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':

    # Run this module to save your config
    write_config_file()
