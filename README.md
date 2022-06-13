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


# Experiments
## Object Detection
Main results on VOC.

|  Model      | Size | AP50 |  Weight  |
|-------------|------|------|----------|
| CCDet-R18   | 640  | 81.0 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r18_81.0.pth) |
| CCDet-R50   | 640  | 84.0 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r50_84.0.pth) |
| CCDet-R101  | 640  | 85.3 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r101_85.3.pth) |

Main results on COCO.

|  Model        |  Backbone     | Size | FPS<sup><br>2080ti | Param | FLOPs |  AP  | AP50 |  Weight  |
|---------------|---------------|------|--------------------|-------|-------|------|------|----------|
| CCDet-R18     | ResNet-18     | 640  |       143          | 20 M  |  24 B | 33.0 | 53.6 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r18_33.0_53.6.pth) |
| CCDet-R50     | ResNet-50     | 640  |        68          | 34 M  |  44 B | 37.7 | 60.0 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r50_38.0_60.0.pth) |
| CCDet-R101    | ResNet-101    | 640  |        46          | 53 M  |  74 B | 40.0 | 61.3 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r101_40.0_61.3.pth)|
| CCDet-E-R18   | ResNet-18     | 640  |                    | 21 M  |  24 B | 34.2 | 53.7 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_e_r18_34.2_53.7.pth) |
| CCDet-E-R50   | ResNet-50     | 640  |                    | 34 M  |  44 B |      |      | [github]() |

AP results on COCO

| Model        |  Scale  |  AP      |  AP50      |  AP75      |  APs      |  APm      |  APl      |
|--------------|---------|----------|------------|------------|-----------|-----------|-----------|
| CCDet-E-R18  |  640    |   34.2   |    53.7    |    36.2    |    17.0   |    37.0   |   47.7    |

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
