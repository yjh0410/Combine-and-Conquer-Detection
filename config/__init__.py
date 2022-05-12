from .dataset_config import d_config
from .model_config import m_config


def build_config(dataset, model_name):
    # build dataset config
    dataset_config = d_config[dataset]
    print(dataset_config)

    # build model config
    model_config = m_config[model_name]
    print(model_config)

    
    return dataset_config, model_config
    