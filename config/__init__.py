from .dataset_config import d_config
from .model_config import m_config


def build_config(dataset, model_name):
    # build dataset config
    if dataset is not None:
        dataset_config = d_config[dataset]
        print(dataset_config)
    else:
        dataset_config == None

    # build model config
    model_config = m_config[model_name]
    print(model_config)

    
    return dataset_config, model_config
    