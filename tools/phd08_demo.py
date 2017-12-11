import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from datasets.factory import get_imdb
from PIL import Image, ImageFont, ImageDraw


# plt.switch_backend('agg')

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(pil_im, image_name, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(os.path.join(cfg.DATA_DIR, 'arialuni.ttf'), 14)

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))],
                       fill=None,
                       outline=(255, 0, 0)
                       )
        draw.text((int(bbox[0]) - 2, int(bbox[1]) - 15),
                  # '{:s} {:.3f}'.format(str(class_name.encode('utf-8')), score),
                  class_name + u' ' + unicode(str(score)),
                  font=font,
                  fill=(0, 0, 255),
                  encoding='utf-8'
                  )
        # cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0), thickness=4)
        # cv2.putText(img=im,
        #             text='{:s} {:.3f}'.format(str(class_name.encode('utf-8')), score),
        #             org=(int(bbox[0]), int(bbox[1])),
        #             fontScale=14,
        #             color=(255,0,0))

    del draw


# def vis_detections(im, class_name, dets,ax, thresh=0.5):
#     """Draw detected bounding boxes."""
#     inds = np.where(dets[:, -1] >= thresh)[0]
#     if len(inds) == 0:
#         return
#
#     for i in inds:
#         bbox = dets[i, :4]
#         score = dets[i, -1]
#
#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1], fill=False,
#                           edgecolor='red', linewidth=3.5)
#             )
#         ax.text(bbox[0], bbox[1] - 2,
#                 '{:s} {:.3f}'.format(class_name, score),
#                 bbox=dict(facecolor='blue', alpha=0.5),
#                 fontsize=14, color='white')
#
#     ax.set_title(('{} detections with '
#                   'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                   thresh),
#                   fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()


def demo(sess, net, image_name, imdb):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    # im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    # fig.savefig(image_name + '.result.png')
    # ax.imshow(im, aspect='equal')

    pil_im = Image.open(im_file)

    classes = imdb.classes

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(pil_im, image_name, cls, dets, ax, thresh=CONF_THRESH)
    plt.close(fig)

    result_file = os.path.join(cfg.DATA_DIR, 'demo', 'result', image_name.split('.')[0] + '_result.jpg')
    # cv2.imwrite(result_file, im)
    pil_im.save(result_file)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--imdb', dest='imdb_name', help='Imdb name',
                        default='phd08_test', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    imdb = get_imdb(args.imdb_name)
    net = get_network(args.demo_net, num_classes=imdb.num_classes)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)

    result_dir = os.path.join(cfg.DATA_DIR, 'demo', 'result')

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    im_names = [str(x) + '.jpg' for x in range(50)]

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name, imdb)

        # plt.show()