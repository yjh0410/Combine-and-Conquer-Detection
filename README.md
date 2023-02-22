# Combine-and-Conquer Detection Framework

## Requirements
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

## Network
- Backbone: [ResNet](https://github.com/yjh0410/FreeYOLO/blob/master/models/backbone/resnet.py)
- Neck: [Dilated Encoder](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/dilated_encoder.py)
- Feature Aggregation: [PaFPN](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/fpn.py)
- Head: [DecoupledHead](https://github.com/yjh0410/FreeYOLO/blob/master/models/head/decoupled_head.py)

## Experiments
### COCO
- Download COCO.
```Shell
cd <FreeYOLOv2_HOME>
cd dataset/scripts/
sh COCO2017.sh
```

- Check COCO
```Shell
cd <FreeYOLOv2_HOME>
python dataset/coco.py
```

**Main results on COCO:**

|  Model      | Size | FPS<sup><br>2080ti |  Param  |  FLOPs  |  AP<sup>val  | AP<sup>test  |    weight     |
|-------------|------|--------------------|---------|---------|--------------|--------------|---------------|
| CCDet-R18   | 640  |     132            | 21.9 M  |  29.5 B |     37.7     |    37.7      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r18_37.7_57.0.pth) |
| CCDet-R50   | 640  |      68            | 36.3 M  |  50.1 B |     41.8     |    41.8      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r50_41.8_61.8.pth) |
| CCDet-R101  | 640  |      45            | 56.3 M  |  81.2 B |     42.6     |    42.6      | [github](https://github.com/yjh0410/Combine-and-Conquer-Detection/releases/download/ccdet_weights/ccdet_r101_42.6_62.5.pth) |


## Train
### Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

### Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.


## Test
```Shell
python test.py -d coco --cuda -v ccdet_r18 -size 640 --weight path/to/weight --show
                  voc            ccdet_r50
                  ...            ...
```


## Eval
```Shell
python eval.py -d coco-val --cuda -v ccdet_r18 -size 640 --weight path/to/weight
                  voc                ccdet_r50
                  ...                ...
```


## Demo
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

## References
If you are using our code, please consider citing our paper.

```
@article{yang2022novel,
  title={A novel fast combine-and-conquer object detector based on only one-level feature map},
  author={Yang, Jianhua and Wang, Ke and Li, Ruifeng and Qin, Zhonghao and Perner, Petra},
  journal={Computer Vision and Image Understanding},
  volume={224},
  pages={103561},
  year={2022},
  publisher={Elsevier}
}
```
