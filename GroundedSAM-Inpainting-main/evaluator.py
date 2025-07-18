# evaluator.py  ── Google Geocode + Haversine
import os, json, math, googlemaps
from pathlib import Path
from typing import Tuple, Dict
from geo_client import analyze_image   # ← 你已有的封装
GOOGLE_MAPS_API_KEY = ""
# ── 1. Google Maps client ───────────────────────────────
GMAPS = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
if not GMAPS.key:
    raise RuntimeError("环境变量 GOOGLE_MAPS_API_KEY 未设置")

def geocode(city: str, country: str) -> Tuple[float, float] | None:
    """返回 (lat, lon)；失败返回 None"""
    q = ", ".join(filter(None, [city, country]))
    resp = GMAPS.geocode(q)
    if resp:
        loc = resp[0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None

# ── 2. Haversine 距离 ───────────────────────────────────
def haversine_km(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    R = 6371.0                           # 地球平均半径 km
    lat1, lon1, lat2, lon2 = map(math.radians, (*p, *q))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

# ── 3. 核心比较函数 ─────────────────────────────────────
def compare(orig_json: str, new_img: str) -> Dict:
    with open(orig_json, "r", encoding="utf-8") as f:
        orig = json.load(f)["response"]["geo_analysis"]

    new, new_json = analyze_image(new_img)      # 第二次分析

    o_top = orig["overall_location_hypothesis"][0]
    n_top = new ["overall_location_hypothesis"][0]

    p = geocode(o_top["city"], o_top["country"])
    q = geocode(n_top["city"], n_top["country"])

    dist_km = haversine_km(p, q) if p and q else None

    return {
        "orig_top": o_top,
        "new_top":  n_top,
        "distance_km": dist_km,
        "new_analysis_json": new_json
    }

# ── 4. 对外入口（run）────────────────────────────────────
def run(original_json: str, new_image_path: str, out_dir: str):
    res = compare(original_json, new_image_path)
    out_file = Path(out_dir) / "eval_result.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False)
    print(f"📏 原→新 直线距离: {res['distance_km']:.1f} km")
    print(f"✅ Evaluation 结果已保存: {out_file}")
