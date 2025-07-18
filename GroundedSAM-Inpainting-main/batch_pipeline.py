#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_pipeline.py —— 批量执行 run_pipeline.py
------------------------------------------------
示例:
  python batch_pipeline.py \
      --img_dir  /data/geodata/imgs \
      --out_root /data/geodata/outputs \
      --dino_cfg path/to/GroundingDINO_SwinT_OGC.py \
      --dino_ckpt groundingdino_swint_ogc.pth \
      --sam_ckpt  sam_vit_h_4b8939.pth \
      --workers 2
"""

import argparse, os, sys, glob, shlex, subprocess, concurrent.futures as cf
from pathlib import Path
from datetime import datetime

RUN_PIPE = Path(__file__).with_name("run_pipeline.py")  # 与本文件同目录

# ────────────────────────────────────────────────────────────────
# 子进程执行的函数必须放模块顶层，才能被 pickle（spawn 模式）
def task(pack):
    """(img_path, cfg_dict) -> (img_path, success_bool)"""
    img_path, cfg = pack
    out_root = Path(cfg["out_root"])

    # 创建该图片专属输出文件夹：<root>/<basename>_YYYYmmdd_HHMMSS/
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(img_path).stem
    out_dir = out_root / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'{sys.executable} "{cfg["run_pipe"]}" '
        f'--image "{img_path}" '
        f'--out_root "{out_dir}" '
        f'--dino_cfg "{cfg["dino_cfg"]}" '
        f'--dino_ckpt "{cfg["dino_ckpt"]}" '
        f'--sam_ckpt "{cfg["sam_ckpt"]}" '
        f'--sam_ver  {cfg["sam_ver"]} '
        f'--device   {cfg["device"]}'
    )

    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode == 0:
        return img_path, True
    else:
        with open(out_dir / "err.log", "w", encoding="utf-8") as fp:
            fp.write(proc.stdout + "\n" + proc.stderr)
        return img_path, False
# ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",   required=True, help="包含待处理图片的目录")
    ap.add_argument("--out_root",  required=True, help="所有输出的根目录")
    ap.add_argument("--dino_cfg",  required=True)
    ap.add_argument("--dino_ckpt", required=True)
    ap.add_argument("--sam_ckpt",  required=True)
    ap.add_argument("--sam_ver",   default="vit_h")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--workers",   type=int, default=1, help="并行进程数 (显存够再开大)")
    args = ap.parse_args()

    # 收集图片路径
    img_paths = sorted(
        glob.glob(os.path.join(args.img_dir, "*.[jp][pn]g")) +   # .jpg .png
        glob.glob(os.path.join(args.img_dir, "*.jpeg"))
    )
    if not img_paths:
        sys.exit("❌ 指定目录下未找到图片")

    print(f"📂 共 {len(img_paths)} 张图片，将输出到 {args.out_root}\n")

    # 打包常量 cfg，避免 pickle 整个 argparse Namespace
    cfg = {
        "out_root": args.out_root,
        "run_pipe": str(RUN_PIPE),
        "dino_cfg": args.dino_cfg,
        "dino_ckpt": args.dino_ckpt,
        "sam_ckpt": args.sam_ckpt,
        "sam_ver": args.sam_ver,
        "device": args.device,
    }

    ok, fail = 0, 0
    if args.workers == 1:
        for img in img_paths:
            _, succ = task((img, cfg))
            ok += succ; fail += (not succ)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for img, succ in ex.map(task, [(p, cfg) for p in img_paths]):
                print("✅" if succ else "❌", Path(img).name)
                ok += succ; fail += (not succ)

    print(f"\n🎉 处理完成！成功 {ok}，失败 {fail}，全部输出位于 {args.out_root}")

if __name__ == "__main__":
    main()
# python batch_pipeline.py --img_dir  /home/wan/geopipeline/GroundedSAM-Inpainting-main/dataset1 --out_root /home/wan/geopipeline/GroundedSAM-Inpainting-main/outputs --dino_cfg /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --dino_ckpt /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/groundingdino_swint_ogc.pth --sam_ckpt /home/wan/geopipeline/GroundedSAM-Inpainting-main/GroundedSAM+Inpainting/Grounded_Segment_Anything/sam_vit_h_4b8939.pth --workers 2