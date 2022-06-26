from .dataset_config import d_config
from .model_config import m_config


def build_config(dataset, version):
    # build dataset config
    if dataset is not None:
        if 'coco' in dataset:
            dataset_config = d_config['coco']
            print(dataset_config)
        elif 'voc' in dataset:
            dataset_config = d_config['voc']
        elif 'widerface' in dataset:
            dataset_config = d_config['widerface']
    else:
        dataset_config == None

    # build model config
    model_config = m_config[version]
    print(model_config)

    
    return dataset_config, model_config
    