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
### CCDet:
Main results on VOC.

|  Model      | Size | FPS | Param(M) | FLOPs(B) | AP50 |  Weight  |
|-------------|------|-----|----------|----------|------|----------|
| CCDet-R18   | 640  | 164 |  20      |  24      | 81.0 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r18_81.0.pth) |
| CCDet-R50   | 640  | 92  |  34      |  44      | 84.0 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r50_84.0.pth) |
| CCDet-R101  | 640  | 63  |  53      |  74      | 85.3 | [github](https://github.com/yjh0410/FreeDet/releases/download/ccdet_weights/ccdet_r101_85.3.pth) |

### CCDet:
Main results on COCO.

|  Model      | Size | FPS | Param(M) | FLOPs(B) |  AP  | AP50 |  Weight  |
|-------------|------|-----|----------|----------|------|------|----------|
| CCDet-R18   | 640  |     |          |          |      |      | [github] |
| CCDet-R50   | 640  |     |          |          |      |      | [github] |
| CCDet-R101  | 640  |     |          |          |      |      | [github] |
| CCDet-D19   | 640  |     |          |          |      |      | [github] |
| CCDet-D53   | 640  |     |          |          |      |      | [github] |
| CCDet-VGG16 | 640  |     |          |          |      |      | [github] |

### CCDet with Modified CSPDarkNet
The Backbone of `YOLOX` is deployed in our `CCDet`.
We also follow its scaling rule to product:
- `CCDet-S` 
- `CCDet-M`
- `CCDet-L`
- `CCDet-X`
- `CCDet-T`
- `CCDet-N`

Main results on COCO.

|  Model      | Size | FPS | Param(M) | FLOPs(B) |  AP  | AP50 |  Weight  |
|-------------|------|-----|----------|----------|------|------|----------|
| CCDet-S     | 640  |     |          |          |      |      | [github] |
| CCDet-M     | 640  |     |          |          |      |      | [github] |
| CCDet-L     | 640  |     |          |          |      |      | [github] |
| CCDet-X     | 640  |     |          |          |      |      | [github] |
| CCDet-T     | 640  |     |          |          |      |      | [github] |
| CCDet-N     | 640  |     |          |          |      |      | [github] |

### CCDet-E:
Main results on COCO.

|  Model       | Size | FPS | Param(M) | FLOPs(B) |  AP  | AP50 |  Weight  |
|--------------|------|-----|----------|----------|------|------|----------|
| CCDet-E-R50  | 640  |     |          |          |      |      | [github] |
| CCDet-E-R101 | 640  |     |          |          |      |      | [github] |

## Face Detection
### CCDet
Main results on WiderFace.

|  Model      | Size | FPS | AP50 |  Weight  |
|-------------|------|-----|------|----------|
| CCDet-R50   | 640  |     |      | [github] |

## Person Detection
### CCDet
Main results on CrowdHuman.

|  Model      | Size | FPS | AP   | MR-2 | JI   |  Weight  |
|-------------|------|-----|------|------|------|----------|
| CCDet-R50   | 640  |     |      |      |      | [github] |


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
