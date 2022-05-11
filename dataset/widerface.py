from __future__ import division , print_function
"""WIDER Face Dataset Classes
author: swordli
"""
import cv2
import random
import numpy as np
import scipy.io
import os.path as osp
import torch.utils.data as data
import matplotlib.pyplot as plt
plt.switch_backend('agg')

try:
    from utils.transforms import  mosaic_augment, mixup_augment
    from utils.label_creator import HMPCreator
except:
    from .utils.transforms import  mosaic_augment, mixup_augment
    from .utils.label_creator import HMPCreator


WIDERFace_CLASSES = ['face']  # always index 0


class WIDERFaceAnnotationTransform(object):
    """Transforms a WIDERFace annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDERFace_CLASSES, range(len(WIDERFace_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords]
        """
        for i in range(len(target)):
            target[i][0] = float(target[i][0]) / width 
            target[i][1] = float(target[i][1]) / height  
            target[i][2] = float(target[i][2]) / width 
            target[i][3] = float(target[i][3]) / height  

            #res.append( [ target[i][0], target[i][1], target[i][2], target[i][3], target[i][4] ] )
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class WIDERFaceDetection(data.Dataset):
    """WIDERFace Detection Dataset Object   
    http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDERFace folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'WIDERFace')
    """

    def __init__(self, 
                 data_root=None,
                 img_size=640,
                 stride=4,
                 image_sets='train',
                 transform=None,
                 color_augment=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0,
                 is_train=False):

        self.data_root = data_root
        self.img_size = img_size
        self.stride = stride
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = WIDERFaceAnnotationTransform()

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

        self.img_ids = list()
        self.label_ids = list()
        self.event_ids = list()

        if self.image_set == 'train':
            path_to_label = osp.join ( self.data_root , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_root , 'WIDER_train/images' )
            fname = "wider_face_train.mat"

        if self.image_set == 'val':
            path_to_label = osp.join ( self.data_root , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_root , 'WIDER_val/images' )
            fname = "wider_face_val.mat"

        if self.image_set == 'test':
            path_to_label = osp.join ( self.data_root , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_root , 'WIDER_test/images' )
            fname = "wider_face_test.mat"

        self.path_to_label = path_to_label
        self.path_to_image = path_to_image
        self.fname = fname
        self.f = scipy.io.loadmat(osp.join(self.path_to_label, self.fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
 
        self._load_widerface()


    def _load_widerface(self):

        error_bbox = 0 
        train_bbox = 0
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]

                if self.image_set in [ 'test' , 'val']:
                    self.img_ids.append( osp.join(self.path_to_image, directory,  im_name + '.jpg') )
                    self.event_ids.append( directory )
                    self.label_ids.append([])
                    continue

                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                bboxes = []
                for i in range(face_bbx.shape[0]):
                    # filter bbox
                    if face_bbx[i][2] < 2 or face_bbx[i][3] < 2 or face_bbx[i][0] < 0 or face_bbx[i][1] < 0:
                        error_bbox +=1
                        #print (face_bbx[i])
                        continue 
                    train_bbox += 1 
                    xmin = float(face_bbx[i][0])
                    ymin = float(face_bbx[i][1])
                    xmax = float(face_bbx[i][2]) + xmin -1 	
                    ymax = float(face_bbx[i][3]) + ymin -1
                    bboxes.append([xmin, ymin, xmax, ymax, 0])

                if ( len(bboxes)==0 ):  #  filter bbox will make bbox none
                    continue
                self.img_ids.append( osp.join(self.path_to_image, directory,  im_name + '.jpg') )
                self.event_ids.append( directory )
                self.label_ids.append( bboxes )
                #yield DATA(os.path.join(self.path_to_image, directory,  im_name + '.jpg'), bboxes)
        print("Error bbox number to filter : %d,  bbox number: %d"  %(error_bbox , train_bbox))
        

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
        return len(self.img_ids)


    def load_image_target(self, index):
        # load a target
        target = self.label_ids[index]
        # load a image
        img = cv2.imread(self.img_ids[index])
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # check target
        if len(target) == 0:
            target = np.zeros([1, 5])
        else:
            target = np.array(target)

        return img, target, height, width


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
                new_index = np.random.randint(0, len(self.img_ids))
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
        return cv2.imread(self.img_ids[index], cv2.IMREAD_COLOR), self.img_ids[index]


    def pull_event(self, index):
        return self.event_ids[index]


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
        img_id = self.img_ids[index]
        anno = self.label_ids[index]
        gt = self.target_transform(anno, 1, 1)
        return img_id.split("/")[-1], gt


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

    dataset = WIDERFaceDetection(
                           data_root='D:\\python_work\\object-detection\\dataset\\WiderFace',
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
