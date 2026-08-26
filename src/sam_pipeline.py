from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import measure


def make_projection_matrix(n_channels):
    """Create the fixed multichannel-to-RGB projection used in the study."""
    W = np.zeros((n_channels, 3), dtype=int)

    for out_channel in range(3):
        row_count = 0
        in_channel = 0
        assigned = 0

        while assigned < n_channels:
            if W[in_channel, :].sum() < 3:
                W[in_channel, out_channel] += 1
                assigned += 1

            row_count += 1
            if row_count == 3:
                row_count = 0
                in_channel += 1
                if in_channel >= n_channels:
                    in_channel = n_channels - 1

    return W / n_channels


def scale_to_uint8(image):
    image = image.astype(np.float32)
    image -= image.min()
    vmax = image.max()
    if vmax > 0:
        image /= vmax
    return (image * 255).astype(np.uint8)


def reduce_channels(image, method, device):
    """Convert an input tile to the 3-channel representation supplied to SAM."""
    if image.ndim == 2:
        image = image[:, :, None]

    _, _, n_channels = image.shape

    if method == "none":
        rgb = scale_to_uint8(image)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.shape[-1] > 3:
            rgb = rgb[:, :, :3]
        return rgb

    if method == "proj":
        W = make_projection_matrix(n_channels)
        rgb = np.einsum("hwc,cd->hwd", image.astype(np.float32), W)
        return scale_to_uint8(rgb)

    if method == "pca":
        x = torch.as_tensor(image, dtype=torch.float32, device=device)
        x = x.reshape(-1, n_channels)
        x = x - x.mean(dim=0, keepdim=True)

        q = min(3, n_channels)
        _, _, V = torch.pca_lowrank(x, q=q)
        pcs = x @ V[:, :q]

        if q < 3:
            pcs = torch.cat([pcs, pcs[:, -1:].repeat(1, 3 - q)], dim=1)

        h, w = image.shape[:2]
        pcs = pcs.reshape(h, w, 3)

        minv = pcs.amin(dim=(0, 1), keepdim=True)
        maxv = pcs.amax(dim=(0, 1), keepdim=True)
        pcs = (pcs - minv) / (maxv - minv + 1e-6)
        return (pcs * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    raise ValueError(f"Unsupported channel method: {method}")


def load_sam(checkpoint, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return SamAutomaticMaskGenerator(model), device


def filter_masks_by_size(masks, image_shape, min_frac, max_frac):
    total_pixels = image_shape[0] * image_shape[1]
    return [
        mask
        for mask in masks
        if min_frac * total_pixels <= mask["area"] <= max_frac * total_pixels
    ]


def masks_to_labels(masks, shape):
    """Convert SAM masks to an integer label image; larger masks are written first."""
    labels = np.zeros(shape, dtype=np.uint16)
    masks = sorted(masks, key=lambda item: item["area"], reverse=True)

    for label_id, mask in enumerate(masks, start=1):
        labels[mask["segmentation"]] = label_id

    return labels


def draw_contours_on_image(image, mask_list):
    output = image.copy()

    for mask in mask_list:
        for contour in measure.find_contours(mask.astype(bool), 0.5):
            contour = np.round(contour).astype(np.int32)
            cv2.polylines(
                output,
                [contour[:, [1, 0]]],
                False,
                (255, 0, 0),
                1,
            )

    return output


def sam_grain_pipeline(
    inputfolder,
    outputfolder,
    channel_method,
    sam_checkpoint,
    sam_model_type,
    run_label=None,
    min_frac=0.0001,
    max_frac=0.33,
):
    """
    Run zero-shot SAM inference for every .npy tile below inputfolder.

    Output naming intentionally matches the MATLAB analysis:
      <outputfolder>/inference_<method>_<model>/.../<stem>_<method>_<model>_labels.mat
      <outputfolder>/inference_<method>_<model>/.../<stem>_<method>_<model>_mask_stack.mat

    Returns one timing record per processed tile.
    """
    inputfolder = Path(inputfolder).resolve()
    outputfolder = Path(outputfolder).resolve()
    sam_checkpoint = Path(sam_checkpoint).resolve()

    if not inputfolder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {inputfolder}")
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {sam_checkpoint}")

    inference_dir = outputfolder / f"inference_{channel_method}_{sam_model_type}"
    inference_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(inputfolder.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy tiles found below {inputfolder}")

    mask_generator, device = load_sam(sam_checkpoint, sam_model_type)

    metadata = {
        "mode": "inference",
        "channel_method": channel_method,
        "sam_model": sam_model_type,
        "inputfolder": str(inputfolder),
        "num_tiles": len(npy_files),
        "device": device,
        "min_frac": min_frac,
        "max_frac": max_frac,
        "run_label": run_label,
    }
    metadata_name = "run_metadata.json" if run_label is None else f"run_metadata_{run_label}.json"
    (inference_dir / metadata_name).write_text(json.dumps(metadata, indent=2))

    timing_records = []

    for index, npy_path in enumerate(npy_files, start=1):
        t0 = time.perf_counter()

        image = np.load(npy_path)
        rgb = reduce_channels(image, channel_method, device)

        masks = mask_generator.generate(rgb)
        masks = filter_masks_by_size(masks, rgb.shape[:2], min_frac, max_frac)

        labels = masks_to_labels(masks, rgb.shape[:2])
        mask_list = [mask["segmentation"] for mask in masks]

        if mask_list:
            mask_stack = np.stack(mask_list, axis=0).astype(np.uint8)
        else:
            h, w = rgb.shape[:2]
            mask_stack = np.zeros((0, h, w), dtype=np.uint8)

        contour_image = draw_contours_on_image(rgb, mask_list)
        binary_mask = (labels > 0).astype(np.uint8) * 255

        relative_path = npy_path.relative_to(inputfolder)
        tile_out = inference_dir / relative_path.parent
        tile_out.mkdir(parents=True, exist_ok=True)

        stem = f"{npy_path.stem}_{channel_method}_{sam_model_type}"

        Image.fromarray(rgb).save(tile_out / f"{stem}_rgb.png")
        Image.fromarray(contour_image).save(tile_out / f"{stem}_contours.png")
        Image.fromarray(binary_mask).save(tile_out / f"{stem}_mask.png")
        np.save(tile_out / f"{stem}_labels.npy", labels)
        np.save(tile_out / f"{stem}_mask_stack.npy", mask_stack)

        savemat(
            tile_out / f"{stem}_labels.mat",
            {"labels": labels},
            do_compression=True,
        )
        savemat(
            tile_out / f"{stem}_mask_stack.mat",
            {"mask_stack": mask_stack},
            do_compression=True,
        )

        elapsed = time.perf_counter() - t0
        timing_records.append(
            {
                "Dataset": run_label or "",
                "input_file": str(relative_path),
                "time_seconds": elapsed,
            }
        )

        print(
            f"[{index:4d}/{len(npy_files):4d}] "
            f"{stem} | {elapsed:.3f} s | {len(mask_list)} masks"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timing_name = (
        "timing_per_image.csv"
        if run_label is None
        else f"timing_per_image_{run_label}.csv"
    )
    write_timing_csv(inference_dir / timing_name, timing_records)

    return timing_records


def write_timing_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Dataset", "input_file", "time_seconds"],
        )
        writer.writeheader()
        writer.writerows(records)
