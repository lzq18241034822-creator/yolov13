# -*- coding: utf-8 -*-
from typing import Dict, List, Any

# 0..6 → 31..37 的疑似降级映射
DEMOTION_MAP = {0:31, 1:32, 2:33, 3:34, 4:35, 5:36, 6:37}

def load_rules_from_cfg(cfg: dict) -> List[dict]:
    re_cfg = cfg.get('stage3_integration', {}).get('rule_engine', {})
    if not re_cfg.get('enabled', False):
        return []
    return re_cfg.get('species_morphology_rules', [])

def check_one(ids5: Dict[str,int], rules: List[dict]) -> Dict[str, Any]:
    """ids5: {'species':g, 'cell_org':g, 'shape':g, 'flagella':g, 'chloroplast':g} 全局ID"""
    sp = ids5['species']; co = ids5['cell_org']; sh = ids5['shape']; fl = ids5['flagella']
    violations=[]
    for r in rules:
        if r.get('species') != sp:
            continue
        req = r.get('required', {})
        forb = r.get('forbidden', {})
        # required：若指定取值不满足，则记录违例
        for k, vals in req.items():
            v = {'cell_org':co, 'shape':sh, 'flagella':fl}.get(k, None)
            if v is not None and v not in vals:
                violations.append(f"required: {k}={v} not in {vals}")
        # forbidden：若命中禁止列表，则记录违例
        for k, vals in forb.items():
            v = {'cell_org':co, 'shape':sh, 'flagella':fl}.get(k, None)
            if v is not None and v in vals:
                violations.append(f"forbidden: {k}={v} in {vals}")
    passed = (len(violations) == 0)
    return {'passed': passed, 'violations': violations}

def apply_demotion_if_needed(ids5: Dict[str,int], rule_result: Dict[str,Any]) -> Dict[str,int]:
    """如果冲突且物种在0..6，则降级到对应不确定类"""
    if rule_result['passed']:
        return ids5
    sp = ids5['species']
    if sp in DEMOTION_MAP:
        ids5 = dict(ids5)  # copy
        ids5['species'] = DEMOTION_MAP[sp]
    return ids5