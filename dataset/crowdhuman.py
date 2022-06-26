import os
import cv2
import json
import random
import numpy as np
import torch
import os.path as osp

try:
    from utils.transforms import  mosaic_augment, mixup_augment
    from utils.label_creator import HMPCreator
except:
    from .utils.transforms import  mosaic_augment, mixup_augment
    from .utils.label_creator import HMPCreator


CrowdHuman_CLASSES = ['person']


# CrowdHuman Detection
class CrowdHumanDetection(torch.utils.data.Dataset):

    def __init__(self, 
                 data_root, 
                 img_size=640, 
                 stride=4,
                 transform=None,
                 color_augment=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0,
                 is_train=False,
                 ignore_label=-1):
        self.data_root = data_root
        self.img_size = img_size
        self.stride = stride
        self.img_folder = os.path.join(data_root, 'Images')
        self.source = os.path.join(data_root, 'annotation_train.odgt') if is_train \
                        else os.path.join(data_root, 'annotation_val.odgt')

        self.records = self.load_json_lines(self.source)
        self.ignore_label = ignore_label

        # augmentation
        self.transform = transform
        self.color_augment = color_augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        if self.mosaic_prob > 0.:
            print('use Mosaic Augmentation ...')
        if self.mixup_prob > 0.:
            print('use Mixup Augmentation ...')

        self.is_train = is_train
        self.gt_creator = HMPCreator(num_classes=20, stride=stride)


    def __getitem__(self, index):
        image, target = self.pull_item(index)

        if self.is_train:
            # create heatmap
            (
                gt_heatmaps, 
                gt_bboxes, 
                gt_bboxes_weights
                ) = self.gt_creator(self.img_size, target)
            target = {
                'gt_heatmaps': gt_heatmaps,
                'gt_bboxes': gt_bboxes,
                'gt_bboxes_weights': gt_bboxes_weights}
        
        return image, target


    def __len__(self):
        return len(self.records)


    def load_json_lines(self, fpath):
        assert os.path.exists(fpath)
        with open(fpath,'r') as fid:
            lines = fid.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        return records


    def load_bbox(self, dict_input, key_name, key_box):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
        bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] == 'person':
                tag = 0
            else:
                tag = -1 # background
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        tag = -1
            # check ttag
            if tag == self.ignore_label:
                continue
            else:
                bbox.append(np.hstack((rb[key_box], tag)))
        
        bboxes = np.vstack(bbox).astype(np.float64)
        # check bboxes
        keep = (bboxes[:, 2]>=0) * (bboxes[:, 3]>=0)
        bboxes = bboxes[keep, :]
        # [x1, y1, bw, bh] -> [x1, y1, x2, y2]
        bboxes[:, 2:4] += bboxes[:, :2]

        return bboxes


    def load_image_target(self, index):
        record = self.records[index]
        # load a image
        image_path = osp.join(self.img_folder, record['ID']+'.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        # load a target
        anno = self.load_bbox(record, 'gtboxes', 'fbox')
        
        # Normalize bbox
        anno[:, [0, 2]] = np.clip(anno[:, [0, 2]], width)
        anno[:, [1, 3]] = np.clip(anno[:, [1, 3]], height)

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }

        return image, target


    def load_mosaic(self, index):
        # load a mosaic image
        ids_list_ = self.records[:index] + self.records[index+1:]
        # random sample other indexs
        id1 = self.records[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        image_list = []
        target_list = []
        # load image and target
        for id_ in ids:
            img_i, target_i = self.load_image_target(id_)
            image_list.append(img_i)
            target_list.append(target_i)

        image, target = mosaic_augment(image_list, target_list, self.img_size)
        
        return image, target


    def pull_item(self, index):
        # load a mosaic image
        if random.random() < self.mosaic_prob:
            image, target = self.load_mosaic(index)

            # MixUp
            if random.random() < self.mixup_prob:
                new_index = np.random.randint(0, len(self.records))
                new_image, new_target = self.load_mosaic(new_index)

                image, target = mixup_augment(image, target, new_image, new_target)

            # augment
            image, target = self.color_augment(image, target)
            
        # load an image and target
        else:
            image, target = self.load_image_target(index)

            # augment
            image, target = self.transform(image, target)

        return image, target


    def pull_image(self, index):
        '''Returns the original image'''
        record = self.records[index]
        # image
        image_path = osp.join(self.img_folder, record['ID']+'.jpg')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return img, record


    def pull_anno(self, index):
        '''Returns the original annotation of image'''
        record = self.records[index]
        # load target
        target = self.load_bbox(record, 'gtboxes', 'fbox', class_names=CrowdHuman_CLASSES)
        
        return target, record


if __name__ == "__main__":
    from utils.transforms import BaseTransforms, TrainTransforms, ValTransforms

    format = 'RGB'
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    img_size = 640
    is_train = False
    trans_config = [{'name': 'DistortTransform',
                     'hue': 0.1,
                     'saturation': 1.5,
                     'exposure': 1.5},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                    {'name': 'ToTensor'},
                    {'name': 'Resize'},
                    {'name': 'Normalize'}]
    transform = TrainTransforms(trans_config=trans_config,
                                img_size=img_size,
                                format=format)

    dataset = CrowdHumanDetection(
                           data_root='/mnt/share/ssd2/dataset/CrowdHuman',
                           img_size=img_size,
                           transform=transform,
                           color_augment=BaseTransforms(),
                           mosaic_prob=0.0,
                           mixup_prob=0.0,
                           is_train=is_train,
                           ignore_label=-1)
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target= dataset[i]
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to BGR format
        if format == 'RGB':
            # denormalize
            image = image * pixel_std + pixel_mean
            image = image[:, :, (2, 1, 0)].astype(np.uint8)
        elif format == 'BGR':
            image = image * pixel_std + pixel_mean
            image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        if is_train:
            # vis heatmap
            gt_heatmaps = target[..., :1].numpy()  

            for i in range(1):
                heatmap = gt_heatmaps[..., i]
                if heatmap.sum() > 0.:
                    heatmap = cv2.resize(heatmap, (img_size, img_size))
                    # [H, W,] -> [H, W, 3]
                    heatmap = np.stack([heatmap]*3, axis=-1)
                    heatmap = 0.4 * image + 0.6 * (heatmap * 255)
                    heatmap = heatmap.astype(np.uint8)
                    # class name
                    cls_name = 'face'
                    cv2.imshow(cls_name, heatmap)
                    cv2.waitKey(0)
                    cv2.destroyWindow(cls_name)
        else:
            boxes = target["boxes"]
            labels = target["labels"]

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = 'face'
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
            cv2.imshow('gt', image)
            # cv2.imwrite(str(i)+'.jpg', img)
            cv2.waitKey(0)
