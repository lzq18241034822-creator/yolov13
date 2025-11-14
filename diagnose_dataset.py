"""
数据集诊断工具：统计50张图的标注分布
输出：每个类别的实例数、每张图的实例数、是否存在空图
"""
import os
import json
import yaml
from pathlib import Path
from collections import Counter
import argparse


def load_yaml(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_unified_txt(txt_path: Path):
    """解析统一TXT，返回实例列表"""
    instances = []
    if not txt_path.exists():
        return instances
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:  # 至少5ID + 1个点(x,y)
                continue
            try:
                species, cell_org, shape, flagella, chloroplast = map(int, parts[:5])
            except ValueError:
                continue
            coords = []
            try:
                coords = list(map(float, parts[5:]))
            except ValueError:
                continue
            if len(coords) < 2 or len(coords) % 2 != 0:
                continue
            instances.append({
                'species': species,
                'cell_org': cell_org,
                'shape': shape,
                'flagella': flagella,
                'chloroplast': chloroplast,
                'num_points': len(coords) // 2
            })
    return instances


def diagnose_dataset(cfg_path: str, data_root: str):
    cfg = load_yaml(cfg_path)
    data_root_p = Path(data_root)

    stats = {
        'total_images': 0,
        'total_instances': 0,
        'species': Counter(),
        'cell_org': Counter(),
        'shape': Counter(),
        'flagella': Counter(),
        'chloroplast': Counter(),
        'instances_per_image': [],
        'empty_images': []
    }

    for split in ['train', 'val']:
        img_dir = data_root_p / 'images' / split
        lbl_dir = data_root_p / 'labels' / split
        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                continue
            stats['total_images'] += 1
            txt_file = lbl_dir / f"{img_file.stem}.txt"

            if not txt_file.exists():
                stats['empty_images'].append(str(img_file))
                stats['instances_per_image'].append(0)
                continue

            instances = parse_unified_txt(txt_file)
            num_inst = len(instances)
            stats['total_instances'] += num_inst
            stats['instances_per_image'].append(num_inst)

            for inst in instances:
                stats['species'][inst['species']] += 1
                stats['cell_org'][inst['cell_org']] += 1
                stats['shape'][inst['shape']] += 1
                stats['flagella'][inst['flagella']] += 1
                stats['chloroplast'][inst['chloroplast']] += 1

    # 输出到控制台
    print("\n" + "=" * 60)
    print("📊 数据集诊断报告（50张图）")
    print("=" * 60)
    print(f"总图像数: {stats['total_images']}")
    print(f"总实例数: {stats['total_instances']}")
    avg_inst = stats['total_instances'] / max(stats['total_images'], 1)
    print(f"平均每张图实例数: {avg_inst:.1f}")
    print(f"空图数量: {len(stats['empty_images'])}")

    # 获取中文名称映射（稳健处理：键转为 int）
    raw_names_zh = cfg.get('classes', {}).get('names_zh', {})
    names_zh = {}
    for k, v in raw_names_zh.items():
        try:
            names_zh[int(k)] = str(v)
        except Exception:
            # 跳过无法解析的键
            continue
    def name_of(gid: int) -> str:
        return names_zh.get(int(gid), f"unknown_{gid}")

    print("\n📌 species 分布:")
    for cls_id, count in stats['species'].most_common():
        name = name_of(cls_id)
        print(f"  {name:20s} (ID={cls_id:2d}): {count:3d} 实例")

    print("\n📌 cell_organization 分布:")
    for cls_id, count in stats['cell_org'].most_common():
        name = name_of(cls_id)
        print(f"  {name:20s} (ID={cls_id:2d}): {count:3d} 实例")

    # 额外：依据 label_spaces 的 to_local 聚合 cell_org 到 4 类（若存在配置）
    cell_org_space = cfg.get('classes', {}).get('label_spaces', {}).get('cell_org', {})
    to_local = cell_org_space.get('to_local', None)
    to_global = cell_org_space.get('to_global', None)
    if isinstance(to_local, dict):
        local_counter = Counter()
        for gid, cnt in stats['cell_org'].items():
            try:
                lid = to_local.get(int(gid), None)
            except Exception:
                lid = None
            if lid is not None:
                local_counter[int(lid)] += cnt
        if local_counter:
            print("\n📎 cell_organization 聚合（训练本地4类）:")
            # 使用 to_global 将本地类映射到对应的全局ID，再取中文名
            for lid, cnt in local_counter.most_common():
                if isinstance(to_global, dict):
                    gid = to_global.get(int(lid), lid)
                else:
                    gid = lid
                name = name_of(gid)
                print(f"  {name:20s} (local={lid}): {cnt:3d} 实例")

    print("\n📌 shape 分布:")
    for cls_id, count in stats['shape'].most_common(10):
        name = name_of(cls_id)
        print(f"  {name:20s} (ID={cls_id:2d}): {count:3d} 实例")

    print("\n📌 flagella 分布:")
    for cls_id, count in stats['flagella'].most_common(10):
        name = name_of(cls_id)
        print(f"  {name:20s} (ID={cls_id:2d}): {count:3d} 实例")

    print("\n📌 chloroplast 分布:")
    for cls_id, count in stats['chloroplast'].most_common(10):
        name = name_of(cls_id)
        print(f"  {name:20s} (ID={cls_id:2d}): {count:3d} 实例")

    print("\n⚠️  建议:")
    if stats['total_instances'] < 200:
        print("  - 实例数<200，建议强数据增强（mosaic/mixup/旋转/翻转）")
    if len(stats['empty_images']) > 0:
        print(f"  - 有 {len(stats['empty_images'])} 张空图，建议检查标注")
    if stats['total_images'] < 100:
        print("  - 图像数<100，Stage2多头分类建议暂时只训练species单头")

    return stats


def main():
    parser = argparse.ArgumentParser(description='统一数据集诊断')
    parser.add_argument('--cfg', type=str, default='g:/yoloV13/µSHM-YOLO/yolov13_transformer_unified_v2_1.yaml')
    parser.add_argument('--data_root', type=str, default='g:/yoloV13/µSHM-YOLO/samples')
    parser.add_argument('--out_json', type=str, default='g:/yoloV13/µSHM-YOLO/tools/reports/dataset_diagnosis.json')
    args = parser.parse_args()

    stats = diagnose_dataset(args.cfg, args.data_root)

    # 保存JSON
    output = {
        'summary': {
            'total_images': stats['total_images'],
            'total_instances': stats['total_instances'],
            'empty_images': stats['empty_images']
        },
        'species': dict(stats['species']),
        'cell_org': dict(stats['cell_org']),
        'shape': dict(stats['shape']),
        'flagella': dict(stats['flagella']),
        'chloroplast': dict(stats['chloroplast'])
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 诊断报告已保存: {args.out_json}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()