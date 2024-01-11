import os

import tensorflow.compat.v1 as tensorflow

from termcolor import cprint
from .sources import utils as F



def get_loader(module_list):
    return list(map(lambda module: (module, tensorflow.train.Saver(
        tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES,
            f'^.*{module}/'))), module_list))


def load_variables_to_session(session, modules_and_variables, checkpoint_path):
    session.run(tensorflow.tables_initializer())
    global_vars = list(map(lambda x: x.name[:-2], tensorflow.get_collection(
        tensorflow.GraphKeys.GLOBAL_VARIABLES)))
    checkpoint_vars = []
    for module, saver in modules_and_variables:
        cprint(module, 'blue')
        load_path = os.path.join(checkpoint_path, module)
        latest_checkpoint = tensorflow.train.latest_checkpoint(load_path)
        print(f'Loading {latest_checkpoint}...')
        assert(latest_checkpoint is not None)

        reader = tensorflow.train.NewCheckpointReader(latest_checkpoint)
        joint_vars = reader.get_variable_to_shape_map()
        #cprint([j for j in joint_vars if 'deep_style_encoder' in j], 'red')
        for var in joint_vars: #global_vars:
            #cprint(var, color='cyan')
            if var == 'global_step': continue
            loaded_var = tensorflow.train.load_variable(latest_checkpoint, var)
            global_var = tensorflow.get_default_graph().get_tensor_by_name(var + ':0')
            if loaded_var.shape != global_var.shape:
                print(f'{var:100s}: checkpoint: {str(loaded_var.shape):20s}'\
                        +f'graph: {str(global_var.shape):20s}')
            checkpoint_vars.append(var)

        saver.restore(session, latest_checkpoint)
        print(f'Successfully loaded the model {latest_checkpoint}')

    print('Only model variables: ')
    F.print_list(list(filter(lambda x: x not in checkpoint_vars, global_vars)), 'red')
    print('Only checkpoint variables: ')
    F.print_list(list(filter(lambda x: x not in global_vars, checkpoint_vars)), 'green')
    print('Saved moving variables: ')
    F.print_list(list(filter(lambda x: ('ema' in x) or ('moving' in x), checkpoint_vars)),
            'blue')
