# Combine-and-Conquer Object Detector
A Simple Baseline for Object Detection.

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
- Backbone: [ResNet](https://github.com/yjh0410/FreeYOLO/blob/master/models/backbone/resnet.py) / [CSPDarkNet53](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/cspdarknet.py)
- Neck: [Dilated Encoder](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/dilated_encoder.py)
- FPN: [YoloPaFPN](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/fpn.py)
- Head: [DecoupledHead](https://github.com/yjh0410/FreeYOLO/blob/master/models/head/decoupled_head.py)

# Experiments
## Object Detection

Main results on COCO.

|  Model      |  Backbone     | Neck |    FPN    | Size | FPS<sup><br>2080ti |  Param  |  FLOPs  |  AP  | AP50 |  Weight  |
|-------------|---------------|------|-----------|------|--------------------|---------|---------|------|------|----------|
| CCDet-R18   | ResNet-18     | DE   | BasicFPN  | 640  |     165            | 21.9 M  |  29.5 B | 34.4 | 54.6 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_e_r18_34.4_54.6.pth) |
| CCDet-R18   | ResNet-18     | DE   | YoloPaFPN | 640  |     132            | 35.3 M  |  49.1 B |      |      | [github]() |
| CCDet-R50   | ResNet-50     | DE   | YoloPaFPN | 640  |                    | 35.3 M  |  49.1 B |      |      | [github]() |
| CCDet-R101  | ResNet-101    | DE   | YoloPaFPN | 640  |                    | 54.3 M  |  79.5 B |      |      | [github]() |
| CCDet-CD53  | CSPDarkNet-53 | DE   | YoloPaFPN | 640  |                    | 37.7 M  |  55.9 B |      |      | [github]() |

AP results on COCO

| Model      |  Scale  |  AP      |  AP50      |  AP75      |  APs      |  APm      |  APl      |
|------------|---------|----------|------------|------------|-----------|-----------|-----------|
| CCDet-R18  |  640    |          |            |            |           |           |           |
| CCDet-R50  |  640    |          |            |            |           |           |           |
| CCDet-R101 |  640    |          |            |            |           |           |           |
| CCDet-CD53 |  640    |          |            |            |           |           |           |

## Face Detection
### CCDet
Main results on WiderFace.

|  Model      | Size | FPS<sup><br>2080ti | AP50 |  Weight  |
|-------------|------|--------------------|------|----------|
| CCDet-R18   | 640  |                    |      | [github] |
| CCDet-R50   | 640  |                    |      | [github] |
| CCDet-R101  | 640  |                    |      | [github] |

## Person Detection
### CCDet
Main results on CrowdHuman.

|  Model      | Size | FPS<sup><br>2080ti |  AP  | MR-2 |  JI  |  Weight  |
|-------------|------|--------------------|------|------|------|----------|
| CCDet-R18   | 640  |                    |      |      |      | [github] |
| CCDet-R50   | 640  |                    |      |      |      | [github] |
| CCDet-R101  | 640  |                    |      |      |      | [github] |

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
