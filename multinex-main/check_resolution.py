import os
from PIL import Image

def get_size(path):
    with Image.open(path) as im:
        w, h = im.size
    return w, h

def compare_folders(input_dir, target_dir):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])
    target_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(exts)])

    print(f"input 文件数: {len(input_files)}")
    print(f"target 文件数: {len(target_files)}\n")

    mismatch = 0
    for f in input_files:
        in_path = os.path.join(input_dir, f)
        gt_path = os.path.join(target_dir, f)

        if not os.path.exists(gt_path):
            print(f"[缺失 GT] {f}")
            mismatch += 1
            continue

        in_w, in_h = get_size(in_path)
        gt_w, gt_h = get_size(gt_path)

        if (in_w, in_h) == (gt_w, gt_h):
            print(f"[一致] {f:35s} input={in_w}x{in_h}  target={gt_w}x{gt_h}")
        else:
            print(f"[不一致] {f:35s} input={in_w}x{in_h}  target={gt_w}x{gt_h}")
            mismatch += 1

    print(f"\n共 {mismatch} 组尺寸不一致")

if __name__ == "__main__":
    compare_folders(
        r"data/MyLowLight/Test/input",
        r"data/MyLowLight/Test/target"
    )