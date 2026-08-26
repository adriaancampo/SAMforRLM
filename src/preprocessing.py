from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def preprocess_dataset(
    input_root,
    method="multichannel",
    gridsize=256,
    gridoverlap=10,
    outputfolder="processed_dataset",
):
    """Tile raw image data into the exact directory structure used downstream."""
    input_root = Path(input_root).resolve()
    outputfolder = Path(outputfolder).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    outputfolder.mkdir(parents=True, exist_ok=True)

    if method == "multichannel":
        _preprocess_multichannel(input_root, gridsize, gridoverlap, outputfolder)

    elif method == "rgb":
        _preprocess_rgb(input_root, gridsize, gridoverlap, outputfolder)

    else:
        raise NotImplementedError(f"Method '{method}' not implemented yet.")


def _get_tile_positions(H, W, gridsize, gridoverlap):
    step = gridsize - gridoverlap

    if step <= 0:
        raise ValueError("gridoverlap must be smaller than gridsize.")

    if H < gridsize or W < gridsize:
        return [], []

    y_positions = list(range(0, H - gridsize + 1, step))
    x_positions = list(range(0, W - gridsize + 1, step))

    return y_positions, x_positions


def _get_red_font(size=24):
    """Return a portable font for diagnostic tile-overview images."""
    for font_name in ("Arial.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_tile_overview(image_array, tile_coords, gridsize, save_path):
    overview = Image.fromarray(image_array)
    draw = ImageDraw.Draw(overview)
    font = _get_red_font(size=24)

    for tile_num, x, y in tile_coords:
        draw.rectangle(
            [x, y, x + gridsize, y + gridsize],
            outline=(255, 0, 0),
            width=3
        )

        draw.text(
            (x + 10, y + 10),
            str(tile_num),
            fill=(255, 0, 0),
            font=font
        )

    overview.save(save_path)


def _normalize_to_uint8(image_array):
    arr = image_array.astype(np.float32)
    arr -= arr.min()

    if arr.max() > 0:
        arr /= arr.max()

    return (arr * 255).astype(np.uint8)


def _preprocess_multichannel(input_root, gridsize, gridoverlap, outputfolder):
    pattern = re.compile(
        r"10(?P<plane>XY|XZ|YZ)_(?P<letter>[a-z])_(?P<angle>\d+)\.tif$",
        re.IGNORECASE
    )

    groups = {}

    for tif in input_root.rglob("*.tif"):
        match = pattern.match(tif.name)
        if match:
            plane = match.group("plane").upper()
            letter = match.group("letter").lower()
            angle = int(match.group("angle"))

            key = (plane, letter)
            groups.setdefault(key, {})[angle] = tif

    print(f"Found {len(groups)} multichannel groups.")

    for (plane, letter), angle_files in groups.items():
        angles = sorted(angle_files.keys())
        print(f"\nProcessing {plane}_{letter} | angles: {angles}")

        imgs = [np.array(Image.open(angle_files[a])) for a in angles]
        stack_full = np.stack(imgs, axis=-1)

        H, W, N = stack_full.shape

        y_positions, x_positions = _get_tile_positions(
            H, W, gridsize, gridoverlap
        )

        if not y_positions or not x_positions:
            print(f"Skipping {plane}_{letter}: image smaller than gridsize.")
            continue

        base_name = f"10{plane}_{letter}.tif"
        base_candidates = list(input_root.rglob(base_name))
        base_img = None

        if base_candidates:
            base_img = np.array(Image.open(base_candidates[0]))
        else:
            print(f"Warning: no base image found for {base_name}")

        group_out = outputfolder / f"{plane}_{letter}"
        stack_out = group_out / "stacks"
        base_out = group_out / "base_tiles"
        overview_out = group_out / "overview"

        stack_out.mkdir(parents=True, exist_ok=True)
        base_out.mkdir(parents=True, exist_ok=True)
        overview_out.mkdir(parents=True, exist_ok=True)

        tile_coords = []
        tile_number = 0

        for y in y_positions:
            for x in x_positions:
                tile_number += 1
                tile_id = f"{plane}_{letter}_tile{tile_number:04d}"

                tile_stack = stack_full[y:y + gridsize, x:x + gridsize, :]
                np.save(stack_out / f"{tile_id}_stack.npy", tile_stack)

                if base_img is not None:
                    base_tile = base_img[y:y + gridsize, x:x + gridsize]
                    Image.fromarray(base_tile).save(
                        base_out / f"{tile_id}_base.tif"
                    )

                tile_coords.append((tile_number, x, y))

        selected_channels = list(range(min(3, N)))
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        for i, ch in enumerate(selected_channels):
            channel = stack_full[:, :, ch].astype(np.float32)
            channel -= channel.min()
            if channel.max() > 0:
                channel /= channel.max()
            rgb[:, :, i] = channel

        if N == 1:
            rgb[:, :, 1] = rgb[:, :, 0]
            rgb[:, :, 2] = rgb[:, :, 0]
        elif N == 2:
            rgb[:, :, 2] = rgb[:, :, 1]

        rgb = (rgb * 255).astype(np.uint8)

        _draw_tile_overview(
            rgb,
            tile_coords,
            gridsize,
            overview_out / f"{plane}_{letter}_overview_tiles.png"
        )

        print(f"Saved {tile_number} tiles for {plane}_{letter}")

    print("\nMultichannel preprocessing done.")


def _preprocess_rgb(input_root, gridsize, gridoverlap, outputfolder):
    rlm_pattern = re.compile(
        r"(?P<sample>[A-Za-z]+_\d+)_RLM\.tif$",
        re.IGNORECASE
    )

    ref_pattern = re.compile(
        r"(?P<sample>[A-Za-z]+_\d+)_Ref\.tif$",
        re.IGNORECASE
    )

    rlm_files = {}
    ref_files = {}

    for tif in input_root.rglob("*.tif"):
        rlm_match = rlm_pattern.match(tif.name)
        ref_match = ref_pattern.match(tif.name)

        if rlm_match:
            rlm_files[rlm_match.group("sample")] = tif

        if ref_match:
            ref_files[ref_match.group("sample")] = tif

    print(f"Found {len(rlm_files)} RGB/RLM files.")

    for sample_id, rlm_path in sorted(rlm_files.items()):
        print(f"\nProcessing {sample_id}")

        rlm_img = np.array(Image.open(rlm_path))

        if rlm_img.ndim == 2:
            rlm_img = np.stack([rlm_img, rlm_img, rlm_img], axis=-1)

        if rlm_img.shape[-1] > 3:
            rlm_img = rlm_img[:, :, :3]

        H, W, C = rlm_img.shape

        y_positions, x_positions = _get_tile_positions(
            H, W, gridsize, gridoverlap
        )

        if not y_positions or not x_positions:
            print(f"Skipping {sample_id}: image smaller than gridsize.")
            continue

        ref_img = None
        if sample_id in ref_files:
            ref_img = np.array(Image.open(ref_files[sample_id]))
        else:
            print(f"Warning: no reference file found for {sample_id}")

        sample_out = outputfolder / sample_id
        rgb_out = sample_out / "rgb_tiles"
        base_out = sample_out / "base_tiles"
        overview_out = sample_out / "overview"

        rgb_out.mkdir(parents=True, exist_ok=True)
        base_out.mkdir(parents=True, exist_ok=True)
        overview_out.mkdir(parents=True, exist_ok=True)

        tile_coords = []
        tile_number = 0

        for y in y_positions:
            for x in x_positions:
                tile_number += 1
                tile_id = f"{sample_id}_tile{tile_number:04d}"

                rgb_tile = rlm_img[y:y + gridsize, x:x + gridsize, :]
                np.save(rgb_out / f"{tile_id}_rgb.npy", rgb_tile)

                if ref_img is not None:
                    ref_tile = ref_img[y:y + gridsize, x:x + gridsize]
                    Image.fromarray(ref_tile).save(
                        base_out / f"{tile_id}_base.tif"
                    )

                tile_coords.append((tile_number, x, y))

        overview_img = rlm_img

        if overview_img.dtype != np.uint8:
            overview_img = _normalize_to_uint8(overview_img)

        _draw_tile_overview(
            overview_img,
            tile_coords,
            gridsize,
            overview_out / f"{sample_id}_overview_tiles.png"
        )

        print(f"Saved {tile_number} RGB tiles for {sample_id}")

    print("\nRGB preprocessing done.")