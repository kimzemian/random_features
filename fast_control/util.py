import configparser
import numpy as np
import toml
import configparser
import pickle, dill


def load_config(path):
    with open(path) as f:
        config = toml.load(f)
    return config


def update_config(config, section, key, value, path="config.toml"):
    config[section][key] = value
    with open(path, "w") as f:
        toml.dump(config, f)


def save_model(model, path):
    serialized_model = dill.dumps(model)
    with open(path, "wb") as file:
        pickle.dump(serialized_model, file)


def load_model(path):
    with open(path, "rb") as file:
        serialized_model = pickle.load(file)
    return dill.loads(serialized_model)
