# geo_client.py
import json, os, re, subprocess, sys, glob, shlex, time
from pathlib import Path
from typing import Dict, Tuple, List

CALL_JSON = "/home/wan/geopipeline/GroundedSAM-Inpainting-main/location attack/call_json.py"          # ← 如果脚本名不同请改这里
PYTHON    = sys.executable          # 当前 Python 解释器

def _run(cmd: str) -> str:
    proc = subprocess.Popen(shlex.split(cmd),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1)
    lines: List[str] = []
    for ln in proc.stdout:
        print(ln, end="")           # 透传
        lines.append(ln)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{CALL_JSON} failed: {cmd}")
    return "".join(lines)

def _latest_json() -> str | None:
    files = glob.glob("analysis_*.json")
    return max(files, key=os.path.getmtime) if files else None

def analyze_image(image_path: str) -> Tuple[Dict, str]:
    """
    调用 call_json.py 生成新的 analysis_*.json，
    返回 (geo_analysis_dict, json_path)
    """
    cmd = f'{PYTHON} "{CALL_JSON}" --image "{image_path}"'
    out = _run(cmd)

    # 从 stdout 捕捉文件名（call_json.py 最后打印 “… analysis_20250712_093201.json”）
    m = re.search(r"(analysis_\d{8}_\d{6}\.json)", out)
    json_path = m.group(1) if m else _latest_json()
    if not json_path or not Path(json_path).exists():
        raise FileNotFoundError("analysis_*.json not found after call_json.py")

    with open(json_path, "r", encoding="utf-8") as fp:
        geo_analysis = json.load(fp)["response"]["geo_analysis"]

    return geo_analysis, json_path
