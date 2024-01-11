import os
import glob
import json
import fire
import importlib

from termcolor import cprint

import tensorflow as tf


def wipe_out(area, case, modules=None, module_depth=0, except_latest=True, print_only=True):
    cprint(modules, color='green')

    def wiper1():
        hyper_params = importlib.import_module(f'model.hyper_params')
        hp = hyper_params.hyper_params
        hp.set_hyper_params_yaml(case, f'./model/{area}/hp.yaml')

        all_list, latests, delete_list = dict(), dict(), dict()

        path = os.path.join('./logs', area, case)

        if isinstance(modules, list):
            saved_modules = modules
        else:
            saved_modules = [p.split('/')[-1] for p in glob.glob(os.path.join(path, *['*']*(int(module_depth) + 1)))]

        print(saved_modules)

        for module in saved_modules:
            module_list = glob.glob(os.path.join(path, module, *['*']*(int(module_depth) + 1)))
            module_list = [file for file in module_list if 'model-' in file]

            all_list[module] = module_list

            try:
                with open(os.path.join(path, module, 'stats.json'), 'r') as f: dictionary = json.load(f)[-1]
                latest = str(dictionary["global_step"])
            except:
                latest = tf.train.latest_checkpoint(checkpoint_dir = os.path.join(path, module, *['*']*(int(module_depth))))# + 1)))
                latest = latest.split('/model-')[-1].split('.')[0]
            latests[module] = latest

            if except_latest:
                delete_modules = [name for name in module_list if name.split('/model-')[-1].split('.')[0] != latest]
            else:
                delete_modules = module_list
            delete_modules.sort()
            delete_list[module] = delete_modules

        cprint(all_list, 'yellow')
        cprint(latests, 'cyan')
        cprint(delete_list, 'red')

        if not print_only:
            size = 0
            for module in saved_modules:
                for name in delete_list[module]:
                    size += os.stat(name).st_size
                    try:
                        cprint('{} - wiped out'.format(name), 'green')
                        os.unlink(name)
                    except:
                        cprint('{} - elimination not available'.format(name), 'red')
            return size
        return 0

    def wiper2():
        # implement when logdir needed to be modified
        return None

    size = wiper1()
    cprint(f'{size/1024/1024/1024:.2f}GB', 'cyan')

if __name__ == '__main__':
    fire.Fire((wipe_out))
