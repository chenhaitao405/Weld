# -*- coding: utf-8 -*-
"""
将 labelme 标注数据整理为 MVTec 兼容目录结构，并支持：
- 缺陷类型（中文）自动翻译为英文目录名
- good/每种缺陷的随机数量上限 (--limit G B)

目录结构示例：
<out_root>/<classname>/
  train|val|test/
    good/
    <anomaly_type_en>/
  ground_truth/
    <anomaly_type_en>/

Created on 2025
"""
import os
import argparse
import shutil
import json
import random
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm


class MVTecPreparer(object):
    """
    从 labelme json 中读取标注：
    - 无有效标注 => 作为 good
    - 有有效标注 => 放到 <anomaly_type_en>/ ，并在 ground_truth/<anomaly_type_en>/ 下生成二值掩码
      * 单一标签 => 使用该标签（翻译后的英文）
      * 多标签 => anomaly = 'mixed'
    """

    # === 中文 -> 英文：覆盖题主列出的 11 类 ===
    LABEL_TRANSLATION = {
        "NONE": "none",
        "伪缺陷": "pseudo_defect",
        "内凹": "concavity",
        "咬边": "undercut",
        "夹渣": "slag_inclusion",
        "夹钨": "tungsten_inclusion",
        "未焊透": "lack_of_penetration",
        "未熔合": "lack_of_fusion",
        "气孔": "porosity",
        "焊瘤": "overlap",
        "裂纹": "crack",
        # 兜底：多标签时使用
        "混合": "mixed",
    }

    def __init__(self, json_dir, out_root=None, classname="custom",
                 split="train", filter_label=None, clean=True, limit=None, seed=None):
        """
        Args:
            json_dir: labelme json 根目录
            out_root: 输出根目录（默认 <json_dir>/MVTecLike）
            classname: MVTec 类别目录名（e.g. 'bottle'）
            split: 'train' | 'val' | 'test'
            filter_label: 需要过滤掉的标签（如 '焊缝'）
            clean: 是否清空已存在的输出类目录
            limit: None 或 (good_limit, anomaly_limit_per_type)
            seed: 随机种子（用于抽样复现）
        """
        self._json_dir = json_dir
        self._classname = classname
        self._split = split
        self._filter_label = filter_label
        self._clean = clean
        self._limit = tuple(limit) if limit is not None else None  # (good_lim, anom_lim)
        self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)

        # 输出目录
        self._out_root = out_root or os.path.join(self._json_dir, "MVTecLike")
        self._class_dir = os.path.join(self._out_root, self._classname)
        self._split_dir = os.path.join(self._class_dir, self._split)
        self._gt_dir = os.path.join(self._class_dir, "ground_truth")

        # 统计
        self._good_count = 0
        self._bad_count = 0
        self._missing_image_count = 0
        self._anomaly_stats = defaultdict(int)

    # ---------- 目录与路径 ----------

    def _make_base_dirs(self):
        """创建基础输出目录（类目录、split 根、ground_truth 根；anomaly 子目录按需创建）"""
        if self._clean and os.path.exists(self._class_dir):
            shutil.rmtree(self._class_dir)
        os.makedirs(self._split_dir, exist_ok=True)
        os.makedirs(os.path.join(self._split_dir, "good"), exist_ok=True)
        os.makedirs(self._gt_dir, exist_ok=True)

    def _ensure_dirs_for_anomaly(self, anomaly_en):
        """确保 split 与 ground_truth 下的异常子目录存在"""
        os.makedirs(os.path.join(self._split_dir, anomaly_en), exist_ok=True)
        os.makedirs(os.path.join(self._gt_dir, anomaly_en), exist_ok=True)

    # ---------- 读取候选 ----------

    def _get_all_json_paths(self):
        """获取所有 json 文件路径（兼容 labels/L|T/train|val 以及平铺）"""
        json_paths = []

        labels_dir = os.path.join(self._json_dir, "labels")
        if os.path.exists(labels_dir):
            for folder in ["L", "T"]:
                folder_path = os.path.join(labels_dir, folder)
                if os.path.exists(folder_path):
                    for sp in ["train", "val"]:
                        split_path = os.path.join(folder_path, sp)
                        if os.path.exists(split_path):
                            for fn in os.listdir(split_path):
                                if fn.endswith(".json"):
                                    json_paths.append(os.path.join(split_path, fn))

        if not json_paths:
            for fn in os.listdir(self._json_dir):
                if fn.endswith(".json"):
                    json_paths.append(os.path.join(self._json_dir, fn))

        return sorted(json_paths)

    def _find_image_path(self, json_path):
        """在若干可能位置查找与 json 同名的图像"""
        base = os.path.splitext(os.path.basename(json_path))[0]
        candidates = [
            json_path.replace(".json", ".jpg").replace("/labels/", "/images/"),
            json_path.replace(".json", ".png").replace("/labels/", "/images/"),
            os.path.join(os.path.dirname(json_path), base + ".jpg"),
            os.path.join(os.path.dirname(json_path), base + ".png"),
            os.path.join(self._json_dir, "images", base + ".jpg"),
            os.path.join(self._json_dir, "images", base + ".png"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ---------- 标注过滤与翻译 ----------

    def _valid_shape(self, shape):
        st = (shape.get("shape_type") or "").lower()
        pts = shape.get("points", [])
        if st == "circle":
            return isinstance(pts, list) and len(pts) >= 2
        elif st == "rectangle":
            return isinstance(pts, list) and len(pts) >= 2
        else:
            return isinstance(pts, list) and len(pts) >= 3

    def _normalize_label(self, label):
        """将原始 label（可能为 None/空/中文）转为英文目录名"""
        if label is None or label == "":
            label = "NONE"
        return self.LABEL_TRANSLATION.get(label, label)

    def _filter_and_collect_shapes(self, json_data):
        """过滤无效/被排除标签，返回 (valid_shapes, label_set_en)"""
        valid_shapes = []
        label_set_en = set()

        for shape in json_data.get("shapes", []):
            raw_label = shape.get("label")
            if raw_label is None or raw_label == "":
                raw_label = "NONE"

            # 过滤指定标签（按原始中文/字符串判断）
            if self._filter_label and raw_label == self._filter_label:
                continue

            if not self._valid_shape(shape):
                continue

            shape = dict(shape)
            shape["shape_type"] = (shape.get("shape_type") or "").lower()
            # 记录英文标签用于目录
            label_en = self._normalize_label(raw_label)

            valid_shapes.append({**shape, "label": label_en})
            label_set_en.add(label_en)

        return valid_shapes, label_set_en

    # ---------- 掩码绘制 ----------

    @staticmethod
    def _draw_polygon(draw, points):
        draw.polygon(points, fill=255)

    @staticmethod
    def _draw_rectangle(draw, points):
        (x1, y1), (x2, y2) = points[0], points[1]
        draw.rectangle([x1, y1, x2, y2], fill=255)

    @staticmethod
    def _draw_circle(draw, points):
        (cx, cy), (px, py) = points[0], points[1]
        import math
        r = math.hypot(px - cx, py - cy)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=255)

    def _rasterize_mask(self, img_w, img_h, shapes):
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        for s in shapes:
            st = s.get("shape_type", "")
            pts = s.get("points", [])
            # 坐标裁剪到图像范围
            clipped = []
            for x, y in pts:
                x = max(0, min(img_w - 1, float(x)))
                y = max(0, min(img_h - 1, float(y)))
                clipped.append((x, y))
            try:
                if st == "rectangle" and len(clipped) >= 2:
                    self._draw_rectangle(draw, clipped)
                elif st == "circle" and len(clipped) >= 2:
                    self._draw_circle(draw, clipped)
                elif len(clipped) >= 3:
                    self._draw_polygon(draw, clipped)
            except Exception as e:
                print(f"[Warn] 绘制 shape 失败: {e}")
        return mask

    # ---------- 主流程 ----------

    def prepare(self):
        print("开始生成 MVTec 目录结构...")
        self._make_base_dirs()

        json_paths = self._get_all_json_paths()
        print(f"找到 {len(json_paths)} 个 JSON 文件")
        if not json_paths:
            print("错误: 未找到任何 JSON")
            return

        # 收集候选
        good_candidates = []  # [(img_path,)]
        defect_candidates = defaultdict(list)  # anomaly_en -> [(img_path, valid_shapes_en, (w,h))]

        for json_path in tqdm(json_paths, desc="扫描候选"):
            json_name = os.path.basename(json_path)

            # 读取 json
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Error] 读取 {json_name} 失败: {e}")
                continue

            # 找图像
            img_path = self._find_image_path(json_path)
            if not img_path or not os.path.exists(img_path):
                print(f"[Warn] 未找到图像: {json_name}")
                self._missing_image_count += 1
                continue

            # 有效标注 & 标签集合（英文）
            valid_shapes, label_set_en = self._filter_and_collect_shapes(data)

            if len(valid_shapes) == 0:
                # 无缺陷 -> good
                good_candidates.append((img_path,))
            else:
                # 有缺陷 -> 单类/混合
                anomaly_en = next(iter(label_set_en)) if len(label_set_en) == 1 else "mixed"
                try:
                    img_w = data["imageWidth"]
                    img_h = data["imageHeight"]
                except KeyError:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                defect_candidates[anomaly_en].append((img_path, valid_shapes, (img_w, img_h)))

        # 抽样限制
        if self._limit is not None:
            good_lim, anom_lim = self._limit
        else:
            good_lim, anom_lim = None, None

        if good_lim is not None and good_lim >= 0 and len(good_candidates) > good_lim:
            good_selected = random.sample(good_candidates, good_lim)
        else:
            good_selected = good_candidates

        defect_selected = {}
        for anom, lst in defect_candidates.items():
            if anom_lim is not None and anom_lim >= 0 and len(lst) > anom_lim:
                defect_selected[anom] = random.sample(lst, anom_lim)
            else:
                defect_selected[anom] = lst

        # ===== 落盘 =====
        # 1) good
        good_dir = os.path.join(self._split_dir, "good")
        for (img_path,) in tqdm(good_selected, desc="保存 good"):
            dst_img = os.path.join(good_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst_img)
            self._good_count += 1

        # 2) 各 anomaly_en 与掩码
        for anom_en, lst in defect_selected.items():
            self._ensure_dirs_for_anomaly(anom_en)
            img_out_dir = os.path.join(self._split_dir, anom_en)
            gt_out_dir = os.path.join(self._gt_dir, anom_en)

            for (img_path, shapes_en, (img_w, img_h)) in tqdm(lst, desc=f"保存 {anom_en}", leave=False):
                # 复制图像
                dst_img = os.path.join(img_out_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst_img)

                # 生成并保存掩码（不论 split，便于可视化；若只需 test，可加判断）
                mask = self._rasterize_mask(img_w, img_h, shapes_en)
                mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                mask.save(os.path.join(gt_out_dir, mask_name))

                self._bad_count += 1
                self._anomaly_stats[anom_en] += 1

        # 统计输出
        print("\n=== 生成完成 ===")
        print(f"good 数量（已保存）: {self._good_count} / 候选 {len(good_candidates)}")
        total_defect_candidates = sum(len(v) for v in defect_candidates.values())
        print(f"缺陷图像数量（已保存）: {self._bad_count} / 候选 {total_defect_candidates}")
        if self._anomaly_stats:
            print("各异常类型计数（英文目录名，已保存）：")
            for k, v in sorted(self._anomaly_stats.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {k}: {v}")
        print(f"缺失图像数: {self._missing_image_count}")
        print(f"输出根目录: {self._out_root}")
        print(f"类目录: {self._class_dir}")
        print(f"数据子目录: {self._split_dir}")
        print(f"掩码目录: {self._gt_dir}")


def build_parser():
    p = argparse.ArgumentParser(
        description="将 labelme 数据整理为 MVTec 兼容结构，支持缺陷类型中文→英文与随机数量上限抽样"
    )
    p.add_argument("--json_dir", type=str, required=True,
                   help="labelme json 所在目录（或包含 labels/L|T/train|val 结构的上级目录）")
    p.add_argument("--out_root", type=str, default=None,
                   help="输出根目录（默认在 json_dir 下创建 MVTecLike）")
    p.add_argument("--classname", type=str, default="custom",
                   help="MVTec 的 <classname> 目录名（例如 'bottle'）")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="输出到哪个划分目录（与读取脚本的 DatasetSplit 对应）")
    p.add_argument("--filter_label", type=str, default="焊缝",
                   help="需要过滤掉的标签名（例如 '焊缝'）")
    p.add_argument("--no_clean", action="store_true",
                   help="若指定，则不删除已存在的输出类目录")
    p.add_argument("--limit", type=int, nargs=2, metavar=("GOOD_LIMIT", "ANOM_LIMIT"),
                   help="数量上限：good 总数 与 每个缺陷类型上限，例如 --limit 2000 100")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子（用于抽样复现实验）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    preparer = MVTecPreparer(
        json_dir=args.json_dir,
        out_root=args.out_root,
        classname=args.classname,
        split=args.split,
        filter_label=args.filter_label,
        clean=(not args.no_clean),
        limit=args.limit,
        seed=args.seed,
    )
    preparer.prepare()
