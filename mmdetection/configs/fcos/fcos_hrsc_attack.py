_base_ = './fcos_r50_caffe_fpn_4x4_1x_coco.py'


model = dict(
    bbox_head=dict(
        _delete_=True,                     # 关键：删除基类的 bbox_head 配置
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)
# 数据集类型和路径
dataset_type = 'CocoDataset'
data_root = '/cloud/cloud-ssd1/collected_files/ljj/APs/TPA-main/results/hrsc_fcos_attack/iter'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,          # 根据显存调整
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= "/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/annotations//instances_train.json",
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes=('ship',)
    ),
    val=dict(
        type=dataset_type,
        ann_file="/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/annotations/instances_val.json",
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=('ship',)
    ),
    test=dict(
        type=dataset_type,
        ann_file='/cloud/cloud-ssd1/collected_files/ljj/dataset/HRSC-coco/annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=('ship',)
    )
)

# 优化器
optimizer = dict(lr=0.001)

# 学习率衰减（24 epoch）
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[16, 22]
)

# 训练周期
runner = dict(type='EpochBasedRunner', max_epochs=24)

# 预训练权重（GN-head 版本）
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
resume_from = None

# 检查点配置：保存验证集上 bbox_mAP 最高的模型
checkpoint_config = dict(interval=1)   # 仅保存每个 epoch 的检查点，不自动选择最佳

