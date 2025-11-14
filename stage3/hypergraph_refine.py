import math
from typing import List, Dict, Tuple, Any


def _center_of_bbox(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bbox_area(b: List[float]) -> float:
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def _safe_id_list(ids: Dict[str, Any], key: str) -> List[int]:
    val = ids.get(key)
    if val is None:
        return []
    if isinstance(val, list):
        return [int(x) for x in val]
    # allow single int
    try:
        return [int(val)]
    except Exception:
        return []


def refine_cell_org_by_proximity(
    detections: List[Dict[str, Any]],
    img_size: Tuple[int, int],
    eps_factor: float = 0.08,
    min_size_px: int = 16,
    id_key_final: str = "ids5_final",
    id_key_raw: str = "ids5_raw",
) -> List[int]:
    """
    Compute refined cell_org id list by clustering cell components using proximity.

    - detections: list of detection dict with keys:
        - bbox_xyxy: [x1,y1,x2,y2]
        - ids5_final / ids5_raw: { 'cell_org': [ids], 'shape': [...], 'species': [...], ... }
    - img_size: (H, W)
    - eps_factor: neighborhood radius as fraction of min(H,W)
    - min_size_px: ignore ROIs with bbox area smaller than this many pixels
    - id_key_final: preferred id container key
    - id_key_raw: fallback id container key

    Returns the refined unique list of cell_org ids for the image.
    """
    H, W = img_size
    scale = max(1.0, min(H, W))
    eps = max(4.0, eps_factor * scale)

    # Collect candidate nodes: each cell-related detection contributes its cell_org ids and center
    nodes: List[Tuple[Tuple[float, float], List[int]]] = []
    for det in detections:
        bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
        if _bbox_area(bbox) < float(min_size_px):
            continue
        center = _center_of_bbox(bbox)
        ids_container: Dict[str, Any] = det.get(id_key_final) or det.get(id_key_raw) or {}
        cell_ids = _safe_id_list(ids_container, "cell_org")
        if not cell_ids:
            # If detection has no cell_org ids, skip
            continue
        nodes.append((center, cell_ids))

    if not nodes:
        return []

    # Build clusters by single-link agglomeration within eps
    clusters: List[List[int]] = []  # list of node indices
    for i, (ci, _) in enumerate(nodes):
        placed = False
        for c in clusters:
            # proximity to any member
            near_any = False
            for j in c:
                cj, _ = nodes[j]
                if _euclidean(ci, cj) <= eps:
                    near_any = True
                    break
            if near_any:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # Merge clusters with transitive closure: if two clusters have members closer than eps, merge
    changed = True
    while changed:
        changed = False
        k = 0
        while k < len(clusters):
            cA = clusters[k]
            merged = False
            m = k + 1
            while m < len(clusters):
                cB = clusters[m]
                close = False
                for ia in cA:
                    ca, _ = nodes[ia]
                    for ib in cB:
                        cb, _ = nodes[ib]
                        if _euclidean(ca, cb) <= eps:
                            close = True
                            break
                    if close:
                        break
                if close:
                    clusters[k] = cA + cB
                    clusters.pop(m)
                    merged = True
                    changed = True
                else:
                    m += 1
            if not merged:
                k += 1

    # Deduplicate ids within each cluster and across clusters
    refined_ids: List[int] = []
    seen: set = set()
    for c in clusters:
        # union of cell_org ids in the cluster
        union_ids: List[int] = []
        for idx in c:
            _, ids = nodes[idx]
            union_ids.extend(ids)
        # stable unique
        for x in union_ids:
            if x not in seen:
                seen.add(x)
                refined_ids.append(x)

    return refined_ids


__all__ = ["refine_cell_org_by_proximity"]