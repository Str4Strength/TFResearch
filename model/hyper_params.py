import yaml
import queue
from pprint import pprint

def load_hyper_params(filename):
    print(filename)
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hp_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hp_dict[k] = v
    return hp_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class HP_dict(Dotdict):

    def __init__(self):
        super(Dotdict, self).__init__()

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    def set_hyper_params_yaml(self, case, hp_file):
        all_hyper_params = load_hyper_params(hp_file)
        hyper_params = all_hyper_params[case]
        all_cases = []
        supers_queue = [case]
        while len(supers_queue) > 0:
            super_case = supers_queue.pop()
            for super in all_hyper_params[super_case]['supers']\
                    if 'supers' in all_hyper_params[super_case] else []:
                if super not in all_cases:
                    all_cases.append(super)
                    supers_queue.insert(0, super)

        for super in all_cases:
            hyper_params = merge_dict(hyper_params, all_hyper_params[super])

        hyper_params = Dotdict(hyper_params)
        for k, v in hyper_params.items():
            setattr(self, k, v)
        pprint(self)
        #self._auto_setting(case)

    def _auto_setting(self, case):
        setattr(self, 'case', case)

        # logdir for a case is automatically set to [logdir_path]/[case]
        setattr(self, 'logdir', '{}/{}'.format(hyper_params.logdir_path, case))
        pprint(self)
        #print('\n'.join([str(item) for item in self.items()]))

hyper_params = HP_dict()
