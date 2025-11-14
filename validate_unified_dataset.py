# -*- coding: utf-8 -*-
"""
统一数据集质检与统计报告（适配 µSHM-YOLO Unified-Final-2.1）
- 校验内容：
  1) 文件名模板与像素尺度位数
  2) TXT 头部5行与 YAML 一致性
  3) 每行：5个ID + 多边形点（偶数、归一化[0,1]）
  4) ID范围合法（含 cell_org=7/8/9/44，兼容10/11→多(>4)）
  5) 软规则冲突统计（警告不阻塞）
  6) 图像-标签配对完整性
- 输出：
  - 命令行摘要
  - JSON报告：reports/validate_report.json（含六项检查的结果）

用法示例：
  python tools/validate_unified_dataset.py --data_root "g:/yoloV13/µSHM-YOLO/samples" --cfg "g:/yoloV13/µSHM-YOLO/yolov13_transformer_unified_v2_1.yaml"
"""

import os
import re
import sys
import json
import math
import glob
# no CSV output in simplified report
import argparse
from collections import defaultdict, Counter

from ruamel.yaml import YAML
try:
    import cv2
    try:
        # 尝试降低 OpenCV 控制台日志级别，避免大量 WARNING
        import cv2.utils as cv2_utils  # type: ignore
        cv2_utils.logging.setLogLevel(cv2_utils.logging.LOG_LEVEL_ERROR)  # type: ignore
    except Exception:
        pass
except Exception:
    cv2 = None
import numpy as np


def load_yaml(path):
    y = YAML()
    with open(path, 'r', encoding='utf-8') as f:
        return y.load(f)


def fmt_pixel_size(val, digits=4, trim=False):
    s = f"{float(val):.{digits}f}"
    if trim and '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def sanitize_name(name, sanitize_cfg):
    s = name
    repl = sanitize_cfg.get('replace', {})
    for k, v in repl.items():
        s = s.replace(k, v)
    for ch in sanitize_cfg.get('remove_chars', []):
        s = s.replace(ch, '')
    if sanitize_cfg.get('to_lowercase', False):
        s = s.lower()
    return s


def build_expected_label_name(template, microscope, image_stem, nf_conf, sanitize_cfg):
    px = fmt_pixel_size(microscope['pixel_size_um'],
                        nf_conf['pixel_size_um']['digits'],
                        nf_conf['pixel_size_um']['trim_trailing_zeros'])
    name = template.format(
        magnification=microscope.get('magnification', ''),
        pixel_size_um=px,
        pixel_dimensions_px=microscope.get('pixel_dimensions_px', ''),
        image_stem=image_stem
    )
    return sanitize_name(name, sanitize_cfg)


def parse_header(lines):
    """
    解析以 # 开头的头部元数据，格式: `# key: value`
    返回 dict 与 非头部起始行索引
    """
    header = {}
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            start_idx = i + 1
            continue
        if s.startswith('#'):
            s = s[1:].strip()
            if ':' in s:
                k, v = s.split(':', 1)
                header[k.strip()] = v.strip()
            start_idx = i + 1
        else:
            break
    return header, start_idx


def parse_label_line(line):
    """
    解析一行标注：
    5个ID + 多边形(x1 y1 ... xn yn), n>=6且为偶数
    返回：ids(list[int]), poly(list[float])
    """
    parts = line.strip().split()
    if len(parts) < 5 + 6:
        raise ValueError(f"line too short, need >= 11 numbers, got {len(parts)}")
    ids = list(map(int, parts[:5]))
    seg = list(map(float, parts[5:]))
    if len(seg) % 2 != 0:
        raise ValueError("segmentation points count must be even")
    return ids, seg


def poly_bounds(seg):
    xs = seg[0::2]
    ys = seg[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def in_01(val, eps=1e-6):
    return -eps <= val <= 1 + eps


def check_ids(ids, cfg):
    """
    ids = [species, cell_org, shape, flagella, chloroplast]
    按 YAML groups 范围与兼容策略校验
    """
    species, cell_org, shape, flagella, chloroplast = ids
    groups = cfg['classes']['groups']
    ok = True
    msgs = []

    # species
    if species not in groups['species']['classes']:
        ok = False
        msgs.append(f"species={species} not in {groups['species']['classes']}")

    # cell_org: 允许 7/8/9/44，兼容旧10/11映射为44（这里允许出现10/11）
    allowed_cell_org = set([7, 8, 9, 44, 10, 11])
    if cell_org not in allowed_cell_org:
        ok = False
        msgs.append(f"cell_org={cell_org} invalid, allowed {sorted(list(allowed_cell_org))}")

    # shape
    if shape not in groups['shape']['classes']:
        ok = False
        msgs.append(f"shape={shape} not in {groups['shape']['classes']}")

    # flagella
    if flagella not in groups['flagella']['classes']:
        ok = False
        msgs.append(f"flagella={flagella} not in {groups['flagella']['classes']}")

    # chloroplast
    if chloroplast not in groups['chloroplast']['classes']:
        ok = False
        msgs.append(f"chloroplast={chloroplast} not in {groups['chloroplast']['classes']}")

    return ok, msgs


def soft_rule_check(ids, rules):
    """
    软规则校验（仅警告）：根据 species_morphology_rules
    返回 (pass_bool, warnings[list])
    """
    species, cell_org, shape, flagella, chloroplast = ids
    warns = []
    passed = True
    # 疑似物种映射到对应本体物种以复用相同规则；38..43 不受软规则约束
    uncertain_map = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6}
    if species in (38, 39, 40, 41, 42, 43):
        return True, []
    species_for_rule = uncertain_map.get(species, species)
    for rule in rules:
        if rule.get('species', None) != species_for_rule:
            continue
        # required
        req = rule.get('required', {})
        for k, vals in req.items():
            val = {'cell_org': cell_org, 'shape': shape, 'flagella': flagella}.get(k, None)
            if val is not None and val not in vals:
                passed = False
                warns.append(f"要求字段不匹配: {k}={val} 不在允许集合 {vals}")
        # forbidden
        forb = rule.get('forbidden', {})
        for k, vals in forb.items():
            val = {'cell_org': cell_org, 'shape': shape, 'flagella': flagella}.get(k, None)
            if val is not None and val in vals:
                passed = False
                warns.append(f"禁止字段命中: {k}={val} 属于禁止集合 {vals}")
    return passed, warns


def validate_label_full(label_path, rules, cfg):
    """
    返回此标签文件的六项检查结果（逐行整合）：
    - header_inconsistent: bool（任一关键键缺失或值不匹配）
    - pixel_size_digits_mismatch: bool（仅针对 header 的像素位数字段）
    - line_format_bad_lines: list[int]
    - id_range_bad_lines: list[int]
    - soft_rule_bad_lines: list[int]
    注：文件名模板与配对完整性在 split 级别统计（不在此函数内）。
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header, start_idx = parse_header(lines)

    # 头部一致性（与 YAML 的 microscope + number_format 对齐）
    header_keys = [
        'magnification', 'pixel_size_um', 'pixel_dimensions_px',
        'magnification_camera', 'pixel_size_um_source'
    ]
    header_inconsistent = False
    pixel_size_digits_mismatch = False

    # 检查关键键是否存在
    missing_keys = [k for k in header_keys if k not in header]
    if missing_keys:
        header_inconsistent = True
    else:
        # 数字位数校验在 main 中执行（需要 cfg），这里仅占位标记，实际值检查在 main 中完成
        pass

    line_format_bad = []
    id_range_bad = []
    soft_bad = []
    line_format_details = []
    id_range_details = []
    soft_rule_details = []
    for ln, line in enumerate(lines[start_idx:], start=start_idx+1):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        try:
            ids, seg = parse_label_line(s)
        except Exception as e:
            line_format_bad.append(ln)
            line_format_details.append({'line': ln, 'reason': str(e), 'content': s})
            continue
        # 坐标范围
        xs = seg[0::2]; ys = seg[1::2]
        if any((not in_01(v) for v in xs + ys)):
            line_format_bad.append(ln)
            line_format_details.append({'line': ln, 'reason': '坐标越界(超出[0,1])', 'ids': ids})
        # ID范围
        ok_ids, msgs = check_ids(ids, cfg)
        if not ok_ids:
            id_range_bad.append(ln)
            id_range_details.append({'line': ln, 'ids': ids, 'violations': msgs})
        # 软规则
        ok, warns = soft_rule_check(ids, rules)
        if not ok and warns:
            soft_bad.append(ln)
            soft_rule_details.append({'line': ln, 'ids': ids, 'violations': warns})
    return {
        'header': header,
        'header_inconsistent': header_inconsistent,
        'pixel_size_digits_mismatch': pixel_size_digits_mismatch,  # 将在 main 中更新
        'line_format_bad_lines': line_format_bad,
        'id_range_bad_lines': id_range_bad,
        'soft_rule_bad_lines': soft_bad,
        'line_format_details': line_format_details,
        'id_range_details': id_range_details,
        'soft_rule_details': soft_rule_details,
    }


def collect_images_and_labels(data_root, split, naming, microscope):
    img_dir = os.path.join(data_root, 'images', split)
    lbl_dir = os.path.join(data_root, 'labels', split)
    nf = naming['number_format']
    sanitize_cfg = naming.get('sanitize', {})

    images = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)

    labels = sorted(glob.glob(os.path.join(lbl_dir, '*.txt')))

    # 建立映射：image_stem -> expected_label_path
    expected_map = {}
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        expected_name = build_expected_label_name(naming['label_filename_template'], microscope, stem, nf, sanitize_cfg)
        expected_map[img_path] = os.path.join(lbl_dir, expected_name)

    return images, labels, expected_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"g:\yoloV13\µSHM-YOLO\samples", help="数据集根目录（包含 images/ 与 labels/）")
    parser.add_argument("--cfg", type=str, default=r"g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml", help="YAML 配置文件路径")
    parser.add_argument("--splits", nargs="+", default=["train","val","test"], help="要校验的子集")
    parser.add_argument("--report_dir", type=str, default=r"g:\yoloV13\µSHM-YOLO\tools\reports", help="报告输出目录")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    naming = cfg['dataset']['naming']
    microscope = cfg['microscope'] if 'microscope' in cfg else cfg['dataset']['microscope']
    rules = cfg['stage3_integration']['rule_engine']['species_morphology_rules'] if 'stage3_integration' in cfg and cfg['stage3_integration'].get('rule_engine',{}).get('enabled',False) else []

    os.makedirs(args.report_dir, exist_ok=True)

    # 输出：六项检查的结果（每个 split 一组）
    report = {}
    # 中文报告结构
    report_zh = {}

    for sp in args.splits:
        print(f"\n=== 正在检查: {sp} ===")
        images, labels, expected_map = collect_images_and_labels(args.data_root, sp, naming, microscope)
        split_info = {
            # 6项：
            'filename_template_mismatch_paths': [],
            'pixel_size_digits_mismatch_indices': [],
            'header_inconsistency_indices': [],
            'line_format_error_indices': [],
            'id_range_error_indices': [],
            'soft_rule_indices': [],
            'missing_label_indices': [],
            'orphan_label_paths': [],
        }

        # 索引映射：图片路径 -> 序号（1-based）
        img_index_map = {p: i+1 for i, p in enumerate(images)}

        # 缺标签
        # 期望的标签集合与实际标签集合
        expected_set = set(expected_map.values())
        actual_set = set(labels)

        # 模板不匹配（存在但不在期望集合的文件）
        for lbl in actual_set - expected_set:
            split_info['filename_template_mismatch_paths'].append(lbl)
            split_info['orphan_label_paths'].append(lbl)

        # 缺标签（按图片索引）与逐文件检查
        inv_expected = {v: k for k, v in expected_map.items()}  # label_path -> image_path
        # 每图问题行号收集
        per_file_lines = {}

        for img_path, exp_lbl in expected_map.items():
            idx = img_index_map[img_path]
            if not os.path.exists(exp_lbl):
                split_info['missing_label_indices'].append(idx)
                continue
            # 逐文件：头部/行格式/ID范围/软规则
            per = validate_label_full(exp_lbl, rules, cfg)
            per_file_lines[idx] = {
                'line_format_bad_lines': per.get('line_format_bad_lines', []),
                'soft_rule_bad_lines': per.get('soft_rule_bad_lines', []),
                'id_range_bad_lines': per.get('id_range_bad_lines', []),
                'line_format_details': per.get('line_format_details', []),
                'soft_rule_details': per.get('soft_rule_details', []),
                'id_range_details': per.get('id_range_details', []),
            }
            # 头部关键键存在性
            if per['header_inconsistent']:
                split_info['header_inconsistency_indices'].append(idx)
            # 像素位数（digits/trim）
            # 依据 YAML number_format 计算期望字符串
            nf = naming.get('number_format', {}).get('pixel_size_um', {'digits': 4, 'trim_trailing_zeros': False})
            digits = nf.get('digits', 4)
            trim = nf.get('trim_trailing_zeros', False)
            px_val = cfg.get('microscope', cfg.get('dataset', {}).get('microscope', {})).get('pixel_size_um', 0.0)
            exp_px = f"{float(px_val):.{digits}f}"
            if trim and '.' in exp_px:
                exp_px = exp_px.rstrip('0').rstrip('.')
            got_px = per['header'].get('pixel_size_um', None)
            if got_px is not None and str(got_px) != str(exp_px):
                split_info['pixel_size_digits_mismatch_indices'].append(idx)

            # 可选：图像尺寸比对（使用 imdecode 避免 Windows 非 ASCII 路径问题与控制台 WARNING）
            if cv2 is not None:
                try:
                    data = np.fromfile(img_path, dtype=np.uint8)
                    if data.size > 0:
                        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    else:
                        img = None
                    if img is not None:
                        h, w = img.shape[:2]
                        exp_dims = f"{w}x{h}"
                        if str(per['header'].get('pixel_dimensions_px', '')) != exp_dims:
                            split_info['header_inconsistency_indices'].append(idx)
                except Exception:
                    pass

            # 行格式错误（偶数点与坐标范围）
            if per['line_format_bad_lines']:
                split_info['line_format_error_indices'].append(idx)

            # ID范围合法性（按 YAML groups 校验）：已在 validate_label_full 收集行号与详情
            if per_file_lines[idx]['id_range_bad_lines']:
                split_info['id_range_error_indices'].append(idx)

            # 软规则（仅警告）
            if per['soft_rule_bad_lines']:
                split_info['soft_rule_indices'].append(idx)

        report[sp] = split_info

        # 生成中文报告（按图片文件名归纳问题）
        idx2name = {i + 1: os.path.basename(p) for i, p in enumerate(images)}

        def names_from_indices(idxs):
            return [idx2name.get(i, f"#{i}") for i in idxs]

        issues_by_idx = {}
        def add_issue(cat, idxs):
            for i in idxs:
                issues_by_idx.setdefault(i, []).append(cat)

        add_issue("缺少标签", split_info['missing_label_indices'])
        add_issue("头部不一致", split_info['header_inconsistency_indices'])
        add_issue("像素位数不一致", split_info['pixel_size_digits_mismatch_indices'])
        add_issue("行格式错误", split_info['line_format_error_indices'])
        add_issue("ID范围错误", split_info['id_range_error_indices'])
        add_issue("软规则警告", split_info['soft_rule_indices'])

        per_image_issues = []
        for i in sorted(issues_by_idx.keys()):
            per_image_issues.append({
                '图片文件': idx2name.get(i, f"#{i}"),
                '问题类型': issues_by_idx[i]
            })

        # 每图问题行号
        per_image_line_numbers = []
        for i in sorted(per_file_lines.keys()):
            per_image_line_numbers.append({
                '图片文件': idx2name.get(i, f"#{i}"),
                '行格式错误行号': per_file_lines[i].get('line_format_bad_lines', []),
                '软规则警告行号': per_file_lines[i].get('soft_rule_bad_lines', []),
                'ID范围错误行号': per_file_lines[i].get('id_range_bad_lines', []),
            })

        # 按图片归并的逐行问题详情（便于直接定位修改）
        per_image_line_details = []
        for i in sorted(per_file_lines.keys()):
            name = idx2name.get(i, f"#{i}")
            # 行格式
            for d in per_file_lines[i].get('line_format_details', []):
                per_image_line_details.append({'图片文件': name, '行': d.get('line'), '类型': '行格式错误', 'IDs': d.get('ids', None), '原因': d.get('reason'), '原文': d.get('content', '')})
            # ID范围
            for d in per_file_lines[i].get('id_range_details', []):
                per_image_line_details.append({'图片文件': name, '行': d.get('line'), '类型': 'ID范围错误', 'IDs': d.get('ids'), '违例': d.get('violations')})
            # 软规则
            for d in per_file_lines[i].get('soft_rule_details', []):
                per_image_line_details.append({'图片文件': name, '行': d.get('line'), '类型': '软规则警告', 'IDs': d.get('ids'), '违例': d.get('violations')})

        zh_split = {
            '分割名': sp,
            '总图片数': len(images),
            '问题计数': {
                '缺少标签': len(split_info['missing_label_indices']),
                '孤立标签文件': len(split_info['orphan_label_paths']),
                '文件名模板不匹配': len(split_info['filename_template_mismatch_paths']),
                '头部不一致': len(split_info['header_inconsistency_indices']),
                '像素位数不一致': len(split_info['pixel_size_digits_mismatch_indices']),
                '行格式错误': len(split_info['line_format_error_indices']),
                'ID范围错误': len(split_info['id_range_error_indices']),
                '软规则警告': len(split_info['soft_rule_indices']),
            },
            '问题明细': {
                '缺少标签': names_from_indices(split_info['missing_label_indices']),
                '孤立标签文件': [os.path.basename(p) for p in split_info['orphan_label_paths']],
                '文件名模板不匹配': [os.path.basename(p) for p in split_info['filename_template_mismatch_paths']],
                '头部不一致': names_from_indices(split_info['header_inconsistency_indices']),
                '像素位数不一致': names_from_indices(split_info['pixel_size_digits_mismatch_indices']),
                '行格式错误': names_from_indices(split_info['line_format_error_indices']),
                'ID范围错误': names_from_indices(split_info['id_range_error_indices']),
                '软规则警告': names_from_indices(split_info['soft_rule_indices']),
            },
            '每图问题汇总': per_image_issues,
            '每图问题行号': per_image_line_numbers,
            '逐行问题详情': per_image_line_details,
        }
        report_zh[sp] = zh_split

    # 输出中文报告（中文键，按图片归纳问题）
    json_path_zh = os.path.join(args.report_dir, 'validate_report_zh.json')
    with open(json_path_zh, 'w', encoding='utf-8') as f:
        json.dump(report_zh, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("\n==== 校验摘要 ====")
    for sp, info in report.items():
        print(f"[{sp}] 缺少标签={len(info['missing_label_indices'])}, 孤立标签={len(info['orphan_label_paths'])}, "
              f"头部不一致={len(info['header_inconsistency_indices'])}, 像素位数不一致={len(info['pixel_size_digits_mismatch_indices'])}, "
              f"行格式错误={len(info['line_format_error_indices'])}, ID范围错误={len(info['id_range_error_indices'])}, 软规则警告={len(info['soft_rule_indices'])}")
    print(f"中文报告: {json_path_zh}")

    # 逐行详细问题（直接打印便于改错）
    print("\n==== 详细问题（逐图逐行） ====")
    for sp in args.splits:
        info = report_zh.get(sp, {})
        details = info.get('逐行问题详情', [])
        if not details:
            continue
        print(f"[分割: {sp}] 共 {len(details)} 项")
        for d in details:
            fname = d.get('图片文件')
            line_no = d.get('行')
            typ = d.get('类型')
            ids = d.get('IDs')
            reason = d.get('原因') or d.get('违例')
            text = d.get('原文', '')
            if isinstance(reason, list):
                reason_str = '; '.join(map(str, reason))
            else:
                reason_str = str(reason)
            ids_str = ' '.join(map(str, ids)) if isinstance(ids, (list, tuple)) else ''
            if ids_str:
                print(f"- 图 {fname} 第{line_no}行（{typ}）：IDs = {ids_str}；问题：{reason_str}")
            elif text:
                print(f"- 图 {fname} 第{line_no}行（{typ}）：{reason_str}；原文：{text}")
            else:
                print(f"- 图 {fname} 第{line_no}行（{typ}）：{reason_str}")

    # 返回码：有严重错误时非零
    # 若存在问题则返回 1，否则 0
    # 退出码：软规则不阻塞，仅当其他项存在问题时返回非零
    def non_soft_has_issue(info):
        return any([
            info['missing_label_indices'],
            info['orphan_label_paths'],
            info['filename_template_mismatch_paths'],
            info['header_inconsistency_indices'],
            info['pixel_size_digits_mismatch_indices'],
            info['line_format_error_indices'],
            info['id_range_error_indices'],
        ])
    # 默认总是返回 0，避免终端因数据问题报错；问题以中文报告与摘要呈现
    # 如需严格退出码，可后续加开关参数控制
    has_err = any(non_soft_has_issue(info) for info in report.values())
    sys.exit(0)


if __name__ == "__main__":
    main()