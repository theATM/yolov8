import os
from pycocotools.coco import COCO

def get_coco_ground_truth(args):
    val_annotate = os.path.join(args.data, args.coco_annots)
    cocoGt = COCO(annotation_file=val_annotate) # , use_ext=True)
    return cocoGt