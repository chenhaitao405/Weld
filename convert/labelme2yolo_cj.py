'''
Created on Aug 18, 2021

@author: xiaosonh
Modified for separate label and image directories
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Labelme2YOLO(object):

    def __init__(self, json_dir, image_dir=None, to_seg=False, filter_label=None, unify_to_crack=False, output_dir=None):
        self._json_dir = json_dir
        # 新增：图像目录参数，如果未提供则使用json目录
        self._image_dir = image_dir if image_dir else json_dir
        self._to_seg = to_seg
        self._filter_label = filter_label  # 添加过滤标签参数
        self._unify_to_crack = unify_to_crack  # 添加统一标签参数

        # 获取标签映射（过滤指定标签）
        self._label_id_map = self._get_label_id_map(self._json_dir)

        # 设置保存路径：如果提供了output_dir则使用，否则使用默认路径
        if output_dir:
            self._save_path_pfx = output_dir
        else:
            i = 'YOLODataset'
            i += '_seg/' if to_seg else '/'
            self._save_path_pfx = os.path.join(self._json_dir, i)

        # 创建输出目录（如果不存在）
        if not os.path.exists(self._save_path_pfx):
            os.makedirs(self._save_path_pfx)

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._save_path_pfx, 'labels/')
        self._image_dir_path = os.path.join(self._save_path_pfx, 'images/')

        for yolo_path in (os.path.join(self._label_dir_path + 'train/'),
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'train/'),
                          os.path.join(self._image_dir_path + 'val/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)

            os.makedirs(yolo_path)

    def _get_label_id_map(self, json_dir):
        # 如果启用了统一标签功能，直接返回只包含crack的映射
        if self._unify_to_crack:
            return OrderedDict([('crack', 0)])

        label_set = set()

        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label = shape['label']
                    # 过滤指定标签
                    if self._filter_label and label == self._filter_label:
                        continue
                    label_set.add(label)

        return OrderedDict([(label, label_id) \
                            for label_id, label in enumerate(label_set)])

    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]

            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]

            return train_json_names, val_json_names

        train_idxs, val_idxs = train_test_split(range(len(json_names)),
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]

        return train_json_names, val_json_names

    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders = [file_name for file_name in os.listdir(self._json_dir) \
                   if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)

        self._make_train_val_dir()

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/'),
                                          (train_json_names, val_json_names)):
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))

                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))

                img_path = self._save_yolo_image(json_name,
                                                 self._image_dir_path,
                                                 target_dir)

                yolo_obj_list = self._get_yolo_object_list(json_data)
                self._save_yolo_label(json_name,
                                      self._label_dir_path,
                                      target_dir,
                                      yolo_obj_list)

        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()


    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print('Converting %s ...' % json_name)

        img_path = self._save_yolo_image(json_name,
                                         self._json_dir, '')

        yolo_obj_list = self._get_yolo_object_list(json_data)
        self._save_yolo_label(json_name, self._json_dir,
                              '', yolo_obj_list)

    def _get_rectangle_shape_yolo_object(self, shape, img_h, img_w):
        """
        将 JSON 中的矩形框 (左上 x1,y1；右下 x2,y2) 转为 YOLO 归一化中心点格式。
        仅处理检测框，不再处理分割/多边形。
        支持以下三种来源：
          1) shape['bbox'] = [x1, y1, x2, y2]
          2) shape['points'] = [[x1, y1], [x2, y2]]
          3) 直接四键：shape['x1'], shape['y1'], shape['x2'], shape['y2']
        返回: (label_id, xc_norm, yc_norm, w_norm, h_norm)
        """
        # 取标签 id（保留你原有的 crack=0 逻辑）
        if self._unify_to_crack:
            label_id = 0
        else:
            label_id = self._label_id_map[shape['label']]

        # 提取 x1,y1,x2,y2
        x1 = y1 = x2 = y2 = None

        if isinstance(shape.get('bbox'), (list, tuple)) and len(shape['bbox']) == 4:
            x1, y1, x2, y2 = map(float, shape['bbox'])
        elif isinstance(shape.get('points'), (list, tuple)) and len(shape['points']) == 2:
            (x1, y1), (x2, y2) = shape['points']
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        elif all(k in shape for k in ('x1', 'y1', 'x2', 'y2')):
            x1, y1, x2, y2 = float(shape['x1']), float(shape['y1']), float(shape['x2']), float(shape['y2'])

        if x1 is None:
            raise ValueError("Invalid rectangle shape: need bbox=[x1,y1,x2,y2], "
                             "or points=[[x1,y1],[x2,y2]], or x1/y1/x2/y2 keys.")

        # 容错：确保次序正确
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # 像素转 YOLO 归一化中心点
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0

        xc_n = round(xc / float(img_w), 6)
        yc_n = round(yc / float(img_h), 6)
        w_n = round(w / float(img_w), 6)
        h_n = round(h / float(img_h), 6)

        # 可选：裁剪到 [0,1]（防止极端越界）
        xc_n = min(max(xc_n, 0.0), 1.0)
        yc_n = min(max(yc_n, 0.0), 1.0)
        w_n = min(max(w_n, 0.0), 1.0)
        h_n = min(max(h_n, 0.0), 1.0)

        return label_id, xc_n, yc_n, w_n, h_n

    def _get_yolo_object_list(self, json_data):
        yolo_obj_list = []

        img_h = json_data['imageHeight']
        img_w = json_data['imageWidth']
        for shape in json_data['shapes']:
            label = shape['label']
            # 过滤指定标签
            if self._filter_label and label == self._filter_label:
                continue
            # 检查points个数，如果小于3则跳过
            # if 'points' in shape and len(shape['points']) < 3:
            #     continue

            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point

            yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list


    def _get_label_id_map_all(self, all_json_paths):
        # 如果启用了统一标签功能，直接返回只包含crack的映射
        if self._unify_to_crack:
            return OrderedDict([('crack', 0)])

        label_set = set()

        for json_path in all_json_paths:
            if os.path.exists(json_path):
                data = json.load(open(json_path))
                for shape in data.get('shapes', []):
                    label = shape.get('label')

                    # 如果label是None或空，转换为字符串"NONE"
                    if label is None or label == '':
                        label = 'NONE'

                    # 过滤指定标签
                    if self._filter_label and label == self._filter_label:
                        continue

                    label_set.add(label)

        return OrderedDict([(label, label_id)
                            for label_id, label in enumerate(sorted(label_set))])


    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        # 如果启用了统一标签，使用'crack'的ID（0），否则使用原标签的ID
        if self._unify_to_crack:
            label_id = 0  # crack的ID总是0
        else:
            label_id = self._label_id_map[shape['label']]

        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                           obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                          for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        # 如果启用了统一标签，使用'crack'的ID（0），否则使用原标签的ID
        if self._unify_to_crack:
            label_id = 0  # crack的ID总是0
        else:
            label_id = self._label_id_map[shape['label']]

        if self._to_seg:
            retval = [label_id]
            for i in shape['points']:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)

            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]

            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        # 移除_convert后缀，保持与原始图像文件名一致
        txt_name = json_name.replace('_convert.json', '.txt').replace('.json', '.txt')
        txt_path = os.path.join(label_dir_path, target_dir, txt_name)

        # 只有当有对象时才写入文件
        if yolo_obj_list:
            with open(txt_path, 'w+') as f:
                for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                    yolo_obj_line = ""
                    for i in yolo_obj:
                        yolo_obj_line += f'{i} '
                    yolo_obj_line = yolo_obj_line[:-1]
                    if yolo_obj_idx != len(yolo_obj_list) - 1:
                        yolo_obj_line += '\n'
                    f.write(yolo_obj_line)
        else:
            # 如果没有对象，创建空文件（或跳过）
            open(txt_path, 'w').close()

    def _save_yolo_image(self, json_name, image_dir_path, target_dir):
        """
        修改后的保存图像函数：
        - JSON文件名格式：第二批_A-001_convert.json
        - 图像文件名格式：第二批_A-001.bmp
        - 需要去掉_convert后缀来匹配图像文件
        """
        json_name_without_ext = os.path.splitext(json_name)[0]

        # 移除_convert后缀来获取图像文件的基础名称
        if json_name_without_ext.endswith('_convert'):
            img_base_name = json_name_without_ext[:-8]  # 移除 '_convert' (8个字符)
        else:
            img_base_name = json_name_without_ext

        # 支持的图片格式
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']

        # 在图像目录中查找对应的图像文件
        src_img_path = None
        img_ext = None
        for ext in image_extensions:
            potential_img_path = os.path.join(self._image_dir, img_base_name + ext)
            if os.path.exists(potential_img_path):
                src_img_path = potential_img_path
                img_ext = ext
                break

        if src_img_path is None:
            print(f"警告: 未找到图像文件: {img_base_name} 在目录 {self._image_dir} (尝试了扩展名: {', '.join(image_extensions)})")
            return None

        # 使用找到的扩展名保存图片
        img_name = img_base_name + img_ext
        dst_img_path = os.path.join(image_dir_path, target_dir, img_name)

        shutil.copy2(src_img_path, dst_img_path)
        print(f"复制图像: {src_img_path} -> {dst_img_path}")
        return dst_img_path


    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._save_path_pfx, 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            # 使用相对路径或绝对路径，取决于需求
            # 这里使用绝对路径以确保YOLO能正确找到数据
            train_path = os.path.abspath(os.path.join(self._image_dir_path, 'train/'))
            val_path = os.path.abspath(os.path.join(self._image_dir_path, 'val/'))

            yaml_file.write('train: %s\n' % train_path)
            yaml_file.write('val: %s\n\n' % val_path)
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))

            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)

        print(f'Dataset.yaml saved to: {yaml_path}')
        print(f'Output directory: {self._save_path_pfx}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, default=None,
                        help='Please input the path of the labelme json files.')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Please input the path of the image files (if different from json_dir).')
    parser.add_argument('--val_size', type=float, nargs='?', default=0.1,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--json_name', type=str, nargs='?', default=None,
                        help='If you put json name, it would convert only one json file to YOLO.')
    parser.add_argument('--seg', action='store_true',
                        help='Convert to YOLOv5 v7.0 segmentation dataset')
    # 添加过滤标签参数
    parser.add_argument('--filter_label', type=str, default=None,
                        help='Label to filter out (e.g., "焊缝")')
    # 添加统一标签参数
    parser.add_argument('--unify_to_crack', action='store_true',
                        help='Unify all labels to "crack" (all defect types will be defined as crack)')
    # 添加输出目录参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for YOLO dataset (default: json_dir/YOLODataset[_seg])')

    parser.add_argument('--mode', type=str, choices=['seg', 'det'], default=None,
                        help='Conversion mode: seg (segmentation) or det (detection). If omitted, falls back to --seg flag for backward compatibility.')

    args = parser.parse_args(sys.argv[1:])

    # 示例使用方式（针对您的特定路径）
    if not args.json_dir:
        # 如果没有提供参数，使用您指定的默认路径
        args.json_dir = '/home/lenovo/code/CHT/datasets/Xray/labelConvert/data_out'
        args.image_dir = '/home/lenovo/code/CHT/datasets/Xray/labelConvert/data'
        print(f"使用默认路径：")
        print(f"  JSON目录: {args.json_dir}")
        print(f"  图像目录: {args.image_dir}")

    convertor = Labelme2YOLO(args.json_dir,
                             image_dir=args.image_dir,
                             to_seg=args.seg,
                             filter_label=args.filter_label,
                             unify_to_crack=args.unify_to_crack,
                             output_dir=args.output_dir)

    convertor.convert(val_size=args.val_size)