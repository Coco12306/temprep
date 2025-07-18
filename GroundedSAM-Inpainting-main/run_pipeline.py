#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============

启动脚本，一键串联：

  1) geo_analyze.py              ——  调用 OpenAI Vision，生成 analysis_*.json
  2) grounded_sam+FLUX.py        ——  读取 analysis_*.json 做目标检测 + 自动 in-painting

依赖：
  * Python 3.9+
  * 事先把 OPENAI_API_KEY 写入环境变量
  * geo_analyze.py / grounded_sam+FLUX.py 位于同级目录（或自行修改常量）

示例：
  python run_pipeline.py ^
      --image D:\location_attack\test_images\building.jpg ^
      --out   outputs ^
      --dino_cfg GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py ^
      --dino_ckpt groundingdino_swint_ogc.pth ^
      --sam_ckpt sam_vit_h_4b8939.pth
"""

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import evaluator

# ---------------------------------------------------------------------------
# 如果脚本名或路径不同，自行修改这里
PYTHON_EXEC   = sys.executable                # 当前解释器
GEO_SCRIPT    = "/home/wan/geopipeline/GroundedSAM-Inpainting-main/location attack/call_json.py"
SEG_SCRIPT    = "/home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/grounded_sam+FLUX.py"
# ---------------------------------------------------------------------------


def run(cmd: str, cwd: str | None = None) -> Tuple[int, str]:
    """
    同步执行子进程，实时把输出透传到终端。
    返回 (return_code, full_stdout_str)。
    """
    print("➤", cmd, "\n")
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
        bufsize=1,
    )
    output_lines: List[str] = []
    for line in proc.stdout:
        print(line, end="")           # 立即打印
        output_lines.append(line)
    proc.wait()
    return proc.returncode, "".join(output_lines)


def find_latest_analysis(search_dir: str = ".", pattern: str = "analysis_*.json") -> str | None:
    """返回 search_dir 下最新的 analysis_*.json 路径（没有则 None）"""
    files = glob.glob(os.path.join(search_dir, pattern))
    return max(files, key=os.path.getmtime) if files else None


def main() -> None:
    parser = argparse.ArgumentParser("Run full geo → segment → inpaint pipeline")
    # 必选
    parser.add_argument("--image", required=True, help="input image path")
    # parser.add_argument("--out",   required=True, help="directory to save final outputs")
    parser.add_argument("--out_root", required=True, help="root dir for outputs")
    parser.add_argument("--dino_cfg",  required=True, help="GroundingDINO config .py")
    parser.add_argument("--dino_ckpt", required=True, help="GroundingDINO checkpoint .pth")
    parser.add_argument("--sam_ckpt",  required=True, help="SAM checkpoint .pth")

    # 可选 & 直传
    parser.add_argument("--sam_ver",   default="vit_h", help="vit_b / vit_l / vit_h (default: vit_h)")
    parser.add_argument("--device",    default="cuda",  help="cuda / cpu (default: cuda)")
    parser.add_argument("--bert_base_uncased_path", default="", help="bert base path if needed")

    # 透传 Grounding-SAM 阈值（可选）
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    # 如果脚本在别的目录
    parser.add_argument("--geo_script", default=GEO_SCRIPT, help="geo_analyze.py path")
    parser.add_argument("--seg_script", default=SEG_SCRIPT, help="grounded_sam+FLUX.py path")

    args = parser.parse_args()

    from datetime import datetime, timezone
    img_stem = Path(args.image).stem
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir  = Path(args.out_root) / f"{img_stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 调 geo_analyze.py ----------
    geo_cmd = (
        f'{PYTHON_EXEC} "{args.geo_script}" '
        f'--image "{args.image}"'
    )
    ret, geo_stdout = run(geo_cmd, cwd=out_dir)
    if ret != 0:
        sys.exit("❌ geo_analyze.py 执行失败")

    # 从 stdout 抓取文件名，若失败则找最新
    m = re.search(r"(analysis_\d{8}_\d{6}\.json)", geo_stdout)
    analysis_json = out_dir / m.group(1) if m else find_latest_analysis(out_dir)
    if not analysis_json or not analysis_json.exists():
        sys.exit("❌ 找不到 analysis_*.json 文件")

    print(f"📑 发现分析结果: {analysis_json}\n")

    # ---------- 2. 调 grounded_sam+FLUX.py ----------
    seg_cmd = (
        f'{PYTHON_EXEC} "{args.seg_script}" '
        f'--analysis_json "{analysis_json}" '
        f'--input_image "{args.image}" '
        # f'--output_dir "{args.out}" '
        f'--output_dir "{out_dir}" '
        f'--config "{args.dino_cfg}" '
        f'--grounded_checkpoint "{args.dino_ckpt}" '
        f'--sam_checkpoint "{args.sam_ckpt}" '
        f'--sam_version "{args.sam_ver}" '
        f'--device "{args.device}" '
        f'--bert_base_uncased_path "{args.bert_base_uncased_path}" '
        f'--box_threshold {args.box_threshold} '
        f'--text_threshold {args.text_threshold}'
    )
    ret2, _ = run(seg_cmd)
    if ret2 != 0:
        sys.exit("❌ grounded_sam+FLUX.py 执行失败")

    print(f"\n✅ inpainting 完成！结果已保存到: {out_dir}")
    # ---------- 3. 调 evaluater.py ----------
    final_img = out_dir / "inpainted_output_final.jpg"
    evaluator.run(original_json=analysis_json,
                new_image_path=str(final_img),
                out_dir=str(out_dir))    


if __name__ == "__main__":
    main()

#   python run.py --image /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/assets/DoleStreet.jpg --out   outputs --dino_cfg /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --dino_ckpt /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/groundingdino_swint_ogc.pth --sam_ckpt /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/sam_vit_h_4b8939.pth