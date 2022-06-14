import argparse
import numpy as np
import cv2
import time
import os
import torch


from config import build_config
from dataset.coco import coco_class_index, coco_class_labels
from dataset.utils.transforms import ValTransforms
from utils.misc import load_weight

from models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Combine-and-Conquer Object Detection')
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='img_size')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='data/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demo/',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('-vs', '--visual_threshold', default=0.4,
                        type=float, help='visual threshold')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visualization results.')
    # model
    parser.add_argument('-v', '--version', default='ccdet_r18', type=str,
                        help='build ccdet')
    parser.add_argument('--weight', default='weights/',
                        type=str, help='Trained state_dict file path to open')
    
    return parser.parse_args()
                    

def plot_bbox_labels(image, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(image, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return image


def visualize(image, bboxes, scores, labels, class_colors, vis_thresh=0.5):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(labels[i])]
            cls_id = coco_class_index[int(labels[i])]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            image = plot_bbox_labels(image, bbox, mess, cls_color, test_scale=ts)

    return image


@torch.no_grad()
def detect(
    model, device, transform, vis_thresh, mode='image', show=False,
    path_to_img=None, path_to_vid=None, path_to_save=None
    ):

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                orig_h, orig_w = frame.shape[:2]
                orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

                # pre-process
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                scores, labels, bboxes = model(x)
                print("Infer: {:.6f} s".format(time.time() - t0))

                # rescale
                bboxes *= orig_size

                # visualization
                frame_processed = visualize(
                    image=frame, 
                    bboxes=bboxes,
                    scores=scores, 
                    labels=labels,
                    class_colors=class_colors,
                    vis_thresh=vis_thresh
                    )
                cv2.imshow('Detection', frame_processed)
                cv2.waitKey(1)

                # To DO:
                # Save the detection video
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            img_file = os.path.join(path_to_img, img_id)
            image = cv2.imread(img_file, cv2.IMREAD_COLOR)

            orig_h, orig_w = image.shape[:2]
            orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

            # pre-process
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            # inference
            scores, labels, bboxes = model(x)
            print("Infer: {:.6f} s".format(time.time() - t0))

            # rescale
            bboxes *= orig_size

            # visualization
            img_processed = visualize(
                image=image, 
                bboxes=bboxes,
                scores=scores, 
                labels=labels,
                class_colors=class_colors,
                vis_thresh=vis_thresh
                )
            cv2.imwrite(os.path.join(save_path, 'images', str(i).zfill(6)+'.jpg'), img_processed)

            if show:
                cv2.imshow('Detection', img_processed)
                cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_path = os.path.join(save_path, cur_time+'.avi')
        fps = 30.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)
        print(save_path)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                orig_h, orig_w = frame.shape[:2]
                orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

                # pre-process
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)

                t0 = time.time()
                # inference
                scores, labels, bboxes = model(x)
                print("Infer: {:.6f} s".format(time.time() - t0))

                # rescale
                bboxes *= orig_size
                
                # visualization
                frame_processed = visualize(
                    image=frame, 
                    bboxes=bboxes,
                    scores=scores, 
                    labels=labels,
                    class_colors=class_colors,
                    vis_thresh=vis_thresh
                    )
                frame_processed_resize = cv2.resize(frame_processed, save_size)

                out.write(frame_processed_resize)
                if show:
                    cv2.imshow('Detection', frame_processed)
                    cv2.waitKey(1)
            else:
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()

    # get device
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg, m_cfg = build_config('coco', args.version)
    
    # build model
    model = build_model(
        model_cfg=m_cfg,
        device=device,
        img_size=args.img_size,
        num_classes=80,
        is_train=False,
        eval_mode=False
        )

    # load trained weight
    model = load_weight(
        device=device, 
        model=model, 
        path_to_ckpt=args.weight
        )

    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # run
    detect(
        model=model,
        device=device,
        transform=transform,
        vis_thresh=args.visual_threshold,
        mode=args.mode,
        show=args.show,
        path_to_img=args.path_to_img,
        path_to_vid=args.path_to_vid,
        path_to_save=args.path_to_save,
        )


if __name__ == '__main__':
    run()
