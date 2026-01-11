# configs/eval/yolo_world_v2_l_coco_val.py

# Use a YOLO-World L config as base (the one already in the repo)
_base_ = '../pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'

# -------------------------
# COCO evaluation settings
# -------------------------
img_scale = (1280, 1280)

# IMPORTANT: inference/test pipeline should NOT load annotations
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)
    ),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts')
    )
]

# COCO val2017 dataset
coco_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco/',
        test_mode=True,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        batch_shapes_cfg=None,
    ),
    class_text_path='data/texts/coco_class_texts_multi_descriptive.json',
    pipeline=test_pipeline
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=coco_val_dataset
)
test_dataloader = val_dataloader

# COCO evaluator
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# Evaluate on 80 COCO classes
model = dict(num_test_classes=80)
