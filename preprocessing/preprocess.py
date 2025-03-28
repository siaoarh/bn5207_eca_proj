import os
import time
import argparse
import SimpleITK as sitk
import numpy as np


def load_image(path):
    return sitk.ReadImage(path)


def check_spacing(img, target_spacing):
    current_spacing = img.GetSpacing()
    return tuple(np.round(current_spacing, 2)) != target_spacing


def resample_image(img, target_spacing, interpolator=sitk.sitkLinear):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetInterpolator(interpolator)
    return resample.Execute(img)


def clip_and_normalize(img, clip_range):
    array = sitk.GetArrayFromImage(img).astype(np.float32)
    array = np.clip(array, clip_range[0], clip_range[1])
    array = (array - clip_range[0]) / (clip_range[1] - clip_range[0])
    new_img = sitk.GetImageFromArray(array)
    new_img.CopyInformation(img)
    return new_img


def save_image(img, output_path, dry_run=False):
    if not dry_run:
        sitk.WriteImage(img, output_path)


def log_message(message, log_path, dry_run=False):
    print(message)
    if not dry_run:
        with open(log_path, "a") as f:
            f.write(message + "\n")


def preprocess_image_and_mask(image_path, output_image_path,
                              mask_path=None, output_mask_path=None,
                              spacing=(1.0, 1.0, 1.0), clip_range=(-100, 400),
                              dry_run=False):
    image = load_image(image_path)

    if check_spacing(image, spacing):
        image = resample_image(image, spacing)

    arr = sitk.GetArrayFromImage(image)
    if arr.min() < 0 or arr.max() > 1.5:
        image = clip_and_normalize(image, clip_range)

    save_image(image, output_image_path, dry_run)

    if mask_path and os.path.exists(mask_path):
        mask = load_image(mask_path)
        if check_spacing(mask, spacing):
            mask = resample_image(mask, spacing, interpolator=sitk.sitkNearestNeighbor)
        save_image(mask, output_mask_path, dry_run)


def batch_preprocess(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mask_dir and args.output_mask_dir:
        os.makedirs(args.output_mask_dir, exist_ok=True)
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, "w") as f:
            f.write("Preprocessing Log\n")

    for filename in os.listdir(args.input_dir):
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            continue

        image_path = os.path.join(args.input_dir, filename)
        output_image_path = os.path.join(args.output_dir, filename)

        base_name = filename.replace(".nii.gz", "").replace(".nii", "")

        # For LiTS17: volume-0.nii → segmentation-0.nii
        if base_name.startswith("volume"):
            mask_base = base_name.replace("volume", "segmentation")
        else:
            mask_base = base_name + "_mask"

        mask_filename = mask_base + ".nii"
        mask_path = os.path.join(args.mask_dir, mask_filename) if args.mask_dir else None
        output_mask_path = os.path.join(args.output_mask_dir, mask_filename) if args.output_mask_dir else None

        start_time = time.time()
        try:
            preprocess_image_and_mask(
                image_path=image_path,
                output_image_path=output_image_path,
                mask_path=mask_path,
                output_mask_path=output_mask_path,
                spacing=tuple(args.spacing),
                clip_range=tuple(args.clip_range),
                dry_run=args.dry_run
            )
            duration = time.time() - start_time
            msg = f"{filename} {'+ mask' if mask_path else ''} -> SUCCESS ({duration:.2f} sec)"
        except Exception as e:
            msg = f"{filename} -> FAILED ({str(e)})"

        log_message(msg, args.log_file, args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess .nii/.nii.gz images and optional masks.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with raw image files")
    parser.add_argument('--mask_dir', type=str, help="Directory with raw mask files (optional)")
    parser.add_argument('--output_dir', type=str, required=True, help="Where to save processed images")
    parser.add_argument('--output_mask_dir', type=str, help="Where to save processed masks (optional)")
    parser.add_argument('--log_file', type=str, default="logs/preprocessing_log.txt", help="Path to log file")
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Target voxel spacing")
    parser.add_argument('--clip_range', type=float, nargs=2, default=(-100, 400), help="HU clipping range")
    parser.add_argument('--dry_run', action='store_true', help="Simulate preprocessing without saving files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_preprocess(args)
