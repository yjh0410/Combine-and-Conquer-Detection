import json
import tempfile
import numpy as np

import torch
from dataset.coco import COCODataset
from .utils import TestTimeAugmentation
    
try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the cocoapi is not installed.")



class COCOEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, 
                 data_dir, 
                 device, 
                 img_size, 
                 testset=False, 
                 transform=None,
                 test_aug=False):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.device = device
        self.testset = testset
        if self.testset:
            image_set = 'test2017'
        else:
            image_set='val2017'

        self.dataset = COCODataset(data_dir=data_dir,
                                   img_size=img_size,
                                   image_set=image_set,
                                   transform=None)

        self.img_size = img_size
        self.transform = transform
        if test_aug:
            print('Use Test Augmentation Trick ...')
            self.test_aug = TestTimeAugmentation(num_classes=80,
                                                 nms_thresh=0.4,
                                                 scale_range=[512, 1280, 128])
        else:
            self.test_aug = None

        self.ap50_95 = -1.
        self.ap50 = -1.


    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            img_h, img_w = img.shape[:2]
            scale = np.array([[img_w, img_h, img_w, img_h]])

            # to tensor
            x = self.transform(img)[0]
            x = x.unsqueeze(0).to(self.device)
            
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                # test augmentation:
                if self.test_aug is not None:
                    scores, labels, bboxes = self.test_aug(x, model)
                else:
                    # inference
                    scores, labels, bboxes = model(x)
                # rescale
                bboxes *= scale
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(labels[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('coco_2017.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_2017.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            return ap50, ap50_95
        else:
            return -1, -1
