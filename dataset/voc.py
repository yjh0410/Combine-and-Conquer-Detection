"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET

try:
    from utils.transforms import  mosaic_augment, mixup_augment
    from utils.label_creator import HMPCreator
except:
    from .utils.transforms import  mosaic_augment, mixup_augment
    from .utils.label_creator import HMPCreator


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 data_root=None,
                 img_size=640,
                 stride=4,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 color_augment=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0,
                 is_train=False):
        self.root = data_root
        self.img_size = img_size
        self.stride = stride
        
        self.image_set = image_sets
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')

        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

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
            target = self.gt_creator(self.img_size, target)
        
        return image, target


    def __len__(self):
        return len(self.ids)


    def load_image_target(self, img_id):
        # load an image
        image = cv2.imread(self._imgpath % img_id)
        height, width, channels = image.shape

        # laod an annotation
        anno = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            anno = self.target_transform(anno)

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
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
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
                new_index = np.random.randint(0, len(self.ids))
                new_image, new_target = self.load_mosaic(new_index)

                image, target = mixup_augment(image, target, new_image, new_target)

            # augment
            image, target = self.color_augment(image, target)
            
        # load an image and target
        else:
            img_id = self.ids[index]
            image, target = self.load_image_target(img_id)

            # augment
            image, target = self.transform(image, target)

        return image, target


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from utils.transforms import BaseTransforms, TrainTransforms, ValTransforms

    format = 'RGB'
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    img_size = 640
    is_train = True
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

    dataset = VOCDetection(data_root='D:\\python_work\\object-detection\\dataset\\VOCdevkit',
                           img_size=img_size,
                           transform=transform,
                           color_augment=BaseTransforms(),
                           mosaic_prob=0.5,
                           mixup_prob=0.5,
                           is_train=is_train)
    
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
            gt_heatmaps = target[..., :20].numpy()     

            for i in range(20):
                heatmap = gt_heatmaps[..., i]
                if heatmap.sum() > 0.:
                    heatmap = cv2.resize(heatmap, (img_size, img_size))
                    # [H, W,] -> [H, W, 3]
                    heatmap = np.stack([heatmap]*3, axis=-1)
                    heatmap = 0.4 * image + 0.6 * (heatmap * 255)
                    heatmap = heatmap.astype(np.uint8)
                    # class name
                    cls_name = VOC_CLASSES[i]
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
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
            cv2.imshow('gt', image)
            # cv2.imwrite(str(i)+'.jpg', img)
            cv2.waitKey(0)
