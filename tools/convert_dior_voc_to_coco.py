"""Convert DIOR VOC-style annotations to COCO JSON.

Usage examples:
  python repro_tools/convert_dior_voc_to_coco.py /path/to/DIOR

  # Custom output location/specs
  python repro_tools/convert_dior_voc_to_coco.py /path/to/DIOR \
    --split trainval:ImageSets/Main/trainval.txt:JPEGImages-trainval:annotations/trainval.json \
    --split test:ImageSets/Main/test.txt:JPEGImages-test:annotations/test.json
"""

import argparse
import json
import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASS2ID = dict((name, idx + 1) for idx, name in enumerate(DIOR_CLASSES))

ANN_SUBDIR_CANDIDATES = (
    'Annotations/Horizontal Bounding Boxes',
    'Annotations',
    'Horizontal Bounding Boxes',
    'Oriented Bounding Boxes',
)


def _read_split_list(split_file):
    with open(split_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_default_splits(root_dir):
    """Resolve default split files for DIOR.

    Priority:
    1) Use ImageSets/Main/trainval.txt + test.txt if present.
    2) Otherwise merge train.txt + val.txt as trainval, and use test.txt.
    """
    main_dir = osp.join(root_dir, 'ImageSets', 'Main')
    trainval_file = osp.join(main_dir, 'trainval.txt')
    test_file = osp.join(main_dir, 'test.txt')

    if osp.isfile(trainval_file) and osp.isfile(test_file):
        return [
            ('trainval', trainval_file, 'JPEGImages-trainval', osp.join(root_dir, 'annotations', 'trainval.json')),
            ('test', test_file, 'JPEGImages-test', osp.join(root_dir, 'annotations', 'test.json')),
        ]

    train_file = osp.join(main_dir, 'train.txt')
    val_file = osp.join(main_dir, 'val.txt')
    if osp.isfile(train_file) and osp.isfile(val_file) and osp.isfile(test_file):
        train_ids = _read_split_list(train_file)
        val_ids = _read_split_list(val_file)
        merged = []
        seen = set()
        for stem in train_ids + val_ids:
            if stem not in seen:
                merged.append(stem)
                seen.add(stem)
        return [
            ('trainval', merged, 'JPEGImages-trainval', osp.join(root_dir, 'annotations', 'trainval.json')),
            ('test', test_file, 'JPEGImages-test', osp.join(root_dir, 'annotations', 'test.json')),
        ]

    raise FileNotFoundError(
        'Cannot find valid split files under ImageSets/Main. '
        'Expected either trainval.txt+test.txt, or train.txt+val.txt+test.txt.')


def _resolve_xml_dir(root_dir, ann_subdir=None):
    if ann_subdir:
        candidate = osp.join(root_dir, ann_subdir)
        if osp.isdir(candidate):
            return candidate
        raise FileNotFoundError('Annotation directory not found: {}'.format(candidate))

    for rel in ANN_SUBDIR_CANDIDATES:
        candidate = osp.join(root_dir, rel)
        if osp.isdir(candidate):
            print('Using annotation directory: {}'.format(candidate))
            return candidate

    raise FileNotFoundError(
        'Cannot find annotation XML directory. Tried: {}'.format(', '.join(ANN_SUBDIR_CANDIDATES)))


def _find_image_name(images_dir, stem):
    for ext in IMG_EXTS:
        candidate = osp.join(images_dir, stem + ext)
        if osp.isfile(candidate):
            return osp.basename(candidate)
    raise FileNotFoundError('Cannot find image file for stem "{}" in {}'.format(stem, images_dir))


def _get_text(node, tag, default=None):
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_xml(xml_path, image_id, file_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        raise ValueError('Missing <size> in {}'.format(xml_path))
    width = int(_get_text(size, 'width', '0'))
    height = int(_get_text(size, 'height', '0'))

    image_info = {
        'id': image_id,
        'file_name': file_name,
        'width': width,
        'height': height,
    }

    annotations = []
    for obj in root.findall('object'):
        name = _get_text(obj, 'name')
        if not name or name not in CLASS2ID:
            continue

        bbox_node = obj.find('bndbox')
        if bbox_node is None:
            continue

        xmin = int(float(_get_text(bbox_node, 'xmin', '0'))) - 1
        ymin = int(float(_get_text(bbox_node, 'ymin', '0'))) - 1
        xmax = int(float(_get_text(bbox_node, 'xmax', '0'))) - 1
        ymax = int(float(_get_text(bbox_node, 'ymax', '0'))) - 1

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, width - 1)
        ymax = min(ymax, height - 1)

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        if bbox_w <= 0 or bbox_h <= 0:
            continue

        difficult_text = _get_text(obj, 'difficult', '0')
        iscrowd = int(difficult_text) if difficult_text is not None else 0

        annotations.append({
            'id': None,
            'image_id': image_id,
            'category_id': CLASS2ID[name],
            'bbox': [float(xmin), float(ymin), float(bbox_w), float(bbox_h)],
            'area': float(bbox_w * bbox_h),
            'iscrowd': iscrowd,
            'segmentation': [[
                float(xmin), float(ymin),
                float(xmax), float(ymin),
                float(xmax), float(ymax),
                float(xmin), float(ymax)
            ]],
        })

    return image_info, annotations


def _convert_split(root_dir, split_name, list_file, images_dir, out_file, xml_dir):
    if isinstance(list_file, list):
        img_stems = list_file
    else:
        if not osp.isfile(list_file):
            raise FileNotFoundError('Split list not found: {}'.format(list_file))
        img_stems = _read_split_list(list_file)
    images = []
    annotations = []
    ann_id = 1

    for image_id, stem in enumerate(img_stems, start=1):
        xml_path = osp.join(xml_dir, stem + '.xml')
        if not osp.isfile(xml_path):
            raise FileNotFoundError('Annotation XML not found: {}'.format(xml_path))

        file_name = _find_image_name(images_dir, stem)
        image_info, sample_annotations = _parse_xml(xml_path, image_id, file_name)
        images.append(image_info)

        for ann in sample_annotations:
            ann['id'] = ann_id
            annotations.append(ann)
            ann_id += 1

    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(DIOR_CLASSES)]
    coco = {'images': images, 'annotations': annotations, 'categories': categories}

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[{}] wrote {} with {} images and {} annotations'.format(
        split_name, out_file, len(images), len(annotations)))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DIOR VOC annotations to COCO JSON')
    parser.add_argument('root_dir', help='Root directory of the raw DIOR dataset')
    parser.add_argument(
        '--ann-subdir',
        default='',
        help=('Relative annotation subdir containing XML files. '
              'Examples: "Annotations/Horizontal Bounding Boxes" or "Annotations". '
              'If omitted, the script auto-detects common DIOR layouts.'))
    parser.add_argument(
        '--split',
        action='append',
        default=[],
        help=('Split spec in the form '
              'name:ImageSets/Main/list.txt:JPEGImages-dir:annotations/out.json. '
              'Can be passed multiple times.'))
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = osp.abspath(args.root_dir)
    xml_dir = _resolve_xml_dir(root_dir, args.ann_subdir.strip() or None)

    if args.split:
        for spec in args.split:
            parts = spec.split(':')
            if len(parts) != 4:
                raise ValueError(
                    'Split spec must be '
                    'name:ImageSets/Main/list.txt:JPEGImages-dir:annotations/out.json, '
                    'but got {}'.format(spec))
            name, list_rel, img_dir_rel, out_rel = parts
            _convert_split(
                root_dir=root_dir,
                split_name=name,
                list_file=osp.join(root_dir, list_rel),
                images_dir=osp.join(root_dir, img_dir_rel),
                out_file=osp.join(root_dir, out_rel),
                xml_dir=xml_dir,
            )
        return

    for name, split_input, img_dir_rel, out_file in _resolve_default_splits(root_dir):
        _convert_split(
            root_dir=root_dir,
            split_name=name,
            list_file=split_input,
            images_dir=osp.join(root_dir, img_dir_rel),
            out_file=out_file,
            xml_dir=xml_dir,
        )


if __name__ == '__main__':
    main()
