# Combine-and-Conquer Detection Framework
A Simple yet Effective Object Detection Framework.

Paper link: Coming soon ...

# Motivation

For a long time, it has been believed that the **Divide-and-Conquer** detection framework is the key to the success of
modern detectors (SSD, RetinaNet, YOLOv4, et al.). Despite the great success, **Divide-and-Conquer** detection framework
has some disadvantages:

- **Divide-and-Conquer** detection framework exacerbates the imbalance between positive and negative samples in the
  training phase. An extremely large number of negative samples are introduced.

- **Divide-and-Conquer** detection framework requires detection head to be deployed on each level of feature, increasing
  model's parameters and FLOPs and impairing the detection speed.

- **Divide-and-Conquer** detection framework suffers from hand-designed hyperparameters for multi-level label assignment,
  such as anchor boxes or scale ranges of interest. However, both of them are empirical and dataset-dependent, impairing
    the model's generalization.

`CornerNet`, `CenterNet` and `YOLOF` have proved that one-level feature map can also achieve excellent performance, but they
did not delve into the advantages of one-level feature, nor did they build a systematic framework for subsequent researchers
to improve the technical framework of one-level.

Therefore, I inherit their ideas and propose a new simple and efficient object detection framework, **Combine-and-Conquer**,
to explore a new efficient detection framework with a one-level feature map. To verity the power of **Combine-and-Conquer**
framework, I design a simple yet effective detector, **CC-Det**. I just deploy existing modules such as `ResNet`, `DilatedEncoder`
and `PaFPN` to construct this simple CC-Det. Designing a most powerful detector is not my purpose and beyond my capabilities.
I just leverage CC-Det to show the effectiveness and potential of this framework. Compared with **Divide-and-Conquer**
framework, it has the following advantages:

- Combine-and-Conquer is a general detection framework. It is modularized into four parts, `Backbone`, `Neck`,
  `Feat Aggregation` and `Detection Head` to make designing new detectors easy.

- I follow the proposed framework to design **CC-Det** detector with a simple network structure and anchor-free
  mechanism. Thanks to the one-level feature, the hassle of carefully designing multi-level label assignment
  like scale range of interest is avoided. Objects of all scales can be detected by this high resolution one-level
  feature. 

- Without any bells and whistles, CC-Det achieves state-of-the-art performance, surpassing previous powerful
  baseline models of multi-level detection framework. In addition to excellent performance, CC-Det has fewer
  model's parameters, lower FLOPs and faster detection speed.
  
- CC-Det is evaluated on multiple datasets, COCO, VOC, WiderFace and CrowdHuman of different detection tasks
  and achieves excellent performance, demonstrating the versatility of the propose detection framework.

Following figure shows a simple example of **Combine-and-Conquer** framework. It looks like Hourglass or CenterNet,
but it is more powerful and fast.

![CCDet with BasicFPN](./img_files/ccdet.png)

Note that `CC-Det` is just a concrete example of **Combine-and-Conquer** framework and has not been carefully
designed. Such a simple model still exhibits strong performance (see the experimental results below.), proving
the potential of the framework. I believe that CC-Det could still be evolved by deploying more novel and powerful
modules. In addition, besides CC-Det, more powerfule and efficient detectors are expected to be designed,
following this framework.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n ccdet python=3.6
```

- Then, activate the environment:
```Shell
conda activate ccdet
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.9.1 and Torchvision >= 0.10.1

# Network
- Backbone: [ResNet](https://github.com/yjh0410/FreeYOLO/blob/master/models/backbone/resnet.py)
- Neck: [Dilated Encoder](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/dilated_encoder.py)
- Feature Aggregation: [PaFPN](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/fpn.py)
- Head: [DecoupledHead](https://github.com/yjh0410/FreeYOLO/blob/master/models/head/decoupled_head.py)

# Experiments
## Object Detection

Main results on COCO.

|  Model      | Size | FPS<sup><br>2080ti |  Param  |  FLOPs  |  AP<sup>val  | AP<sup>test  |    weight     |
|-------------|------|--------------------|---------|---------|--------------|--------------|---------------|
| CCDet-R18   | 640  |     132            | 21.9 M  |  29.5 B |     37.7     |    37.7      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r18_37.7_57.0.pth) |
| CCDet-R50   | 640  |      68            | 36.3 M  |  50.1 B |     41.8     |    41.8      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r50_41.8_61.8.pth) |
| CCDet-R101  | 640  |      45            | 56.3 M  |  81.2 B |     42.6     |    42.6      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r101_42.6_62.5.pth) |

# Dataset
## MSCOCO Dataset
### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


# Train
## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

## Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.


# Test
```Shell
python test.py -d coco --cuda -v ccdet_r18 -size 640 --weight path/to/weight --show
                  voc            ccdet_r50
                  ...            ...
```


# Eval
```Shell
python eval.py -d coco-val --cuda -v ccdet_r18 -size 640 --weight path/to/weight
                  voc                ccdet_r50
                  ...                ...
```


# Demo
I have provide some images in `data/demo/images/`, 
so you can run following command to run a demo:

```Shell
python demo.py --cuda \
               --mode image \
               --path_to_img data/demo/images/ \
               -v ccdet_r18 \
               -size 640 \
               --weight path/to/weight
```

If you want run a demo of streaming video detection, 
you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --cuda \
               --mode video \
               --path_to_img data/demo/videos/video_file \
               -v ccdet_r18 \
               -size 640 \
               --weight path/to/weight
```

If you want run video detection with your camera, 
you need to set `--mode` to `camera`。

```Shell
python demo.py --cuda \
               --mode camera \
               -v ccdet_r18 \
               -size 640 \
               --weight path/to/weight
```
