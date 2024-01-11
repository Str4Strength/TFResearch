import os
import sys
import fire
import time
import itertools
import importlib

import numpy as np
import tensorflow as tf


def test(model, case, gpu=None, **kwargs):
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit=true"
    assert os.path.exists("model/test")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu))
    while model.endswith("/"): model.endswith("/", ".")

    hyper_params = importlib.import_module("model.hyper_params")
    hp = hyper_params.hyper_params
    hp.set_hyper_params_yaml(case, f"./model/{model}/hp.yaml")
    importlib.import_module(f"model.test.{model}").test(case, gpu, **kwargs)

if __name__ == "__main__": fire.Fire(test)

