import argparse
import os
import sys
import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def extract_chip(image_pil, bbox_xyxy, margin_percent=0.15):
    """
    Extract a chip (crop) from the image with a context margin.

    Args:
        image_pil: PIL Image object
        bbox_xyxy: [x0, y0, x1, y1] in pixel coordinates
        margin_percent: Percentage of bbox size to add as margin (0.15 = 15%)

    Returns:
        chip: Cropped PIL Image
        chip_bbox: [x0, y0, x1, y1] actual crop coordinates used
    """
    W, H = image_pil.size
    x0, y0, x1, y1 = bbox_xyxy

    # Calculate box dimensions
    box_width = x1 - x0
    box_height = y1 - y0

    # Calculate margin in pixels
    margin_x = int(box_width * margin_percent)
    margin_y = int(box_height * margin_percent)

    # Expand bbox with margin, but clip to image bounds
    chip_x0 = max(0, x0 - margin_x)
    chip_y0 = max(0, y0 - margin_y)
    chip_x1 = min(W, x1 + margin_x)
    chip_y1 = min(H, y1 + margin_y)

    # Crop the image
    chip = image_pil.crop((chip_x0, chip_y0, chip_x1, chip_y1))

    return chip, [chip_x0, chip_y0, chip_x1, chip_y1]


def calculate_chip_quality_metrics(chip):
    """
    Calculate basic quality metrics for a chip.

    Args:
        chip: PIL Image

    Returns:
        dict with quality metrics
    """
    # Convert to numpy array
    chip_np = np.array(chip.convert('L'))  # Convert to grayscale for analysis

    # Sharpness metric (variance of Laplacian)
    from scipy import ndimage
    laplacian = ndimage.laplace(chip_np)
    sharpness = laplacian.var()

    # Exposure metrics
    mean_intensity = chip_np.mean()
    std_intensity = chip_np.std()

    # Simple contrast metric
    contrast = chip_np.max() - chip_np.min()

    # Convert numpy types to Python native types for JSON serialization
    return {
        'sharpness': float(sharpness),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'contrast': int(contrast),
        'is_blurry': bool(sharpness < 100),  # Convert numpy bool to Python bool
        'is_overexposed': bool(mean_intensity > 240),
        'is_underexposed': bool(mean_intensity < 15),
        'is_low_contrast': bool(contrast < 50)
    }


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def create_thumbnail_with_masks(image_pil, detections, size=384):
    """
    Create a thumbnail of the image with detection masks overlaid.

    Args:
        image_pil: Original PIL Image
        detections: List of detection dicts with bbox_xyxy
        size: Target size for thumbnail (will maintain aspect ratio)

    Returns:
        thumbnail: PIL Image with masks overlaid
    """
    # Create a copy and resize
    thumbnail = image_pil.copy()
    thumbnail.thumbnail((size, size), Image.Resampling.LANCZOS)

    # Calculate scaling factor
    scale_x = thumbnail.width / image_pil.width
    scale_y = thumbnail.height / image_pil.height

    # Draw masks on thumbnail
    draw = ImageDraw.Draw(thumbnail, 'RGBA')

    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det['bbox_xyxy']
        # Scale coordinates
        x0_scaled = int(x0 * scale_x)
        y0_scaled = int(y0 * scale_y)
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)

        # Use different colors for each detection
        color = tuple(np.random.RandomState(i).randint(100, 255, size=3).tolist())
        # Semi-transparent overlay
        draw.rectangle([x0_scaled, y0_scaled, x1_scaled, y1_scaled],
                       outline=color + (255,), width=2)
        draw.rectangle([x0_scaled, y0_scaled, x1_scaled, y1_scaled],
                       fill=color + (50,))

    return thumbnail


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    if not cpu_only and torch.backends.mps.is_available():
        args.device = "mps"
    elif not cpu_only and torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False,
                         token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    if not cpu_only and torch.backends.mps.is_available():
        device = "mps"
    elif not cpu_only and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Running on device:", device)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device)  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
    "The positions of start and end positions of phrases of interest. \
    For example, a caption is 'a cat and a dog', \
    if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
    if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
    ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")

    # New arguments for chip extraction
    parser.add_argument("--extract-chips", action="store_true", help="Extract and save detection chips")
    parser.add_argument("--chip-margin", type=float, default=0.15, help="Context margin for chips (0.15 = 15%)")
    parser.add_argument("--chip-quality", action="store_true", help="Calculate quality metrics for chips")
    parser.add_argument("--create-thumbnail", action="store_true", help="Create thumbnail with masks overlay")
    parser.add_argument("--thumbnail-size", type=int, default=384, help="Size for thumbnail with masks")

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # Create chips subdirectory if extracting chips
    if args.extract_chips:
        chips_dir = os.path.join(output_dir, "chips")
        os.makedirs(chips_dir, exist_ok=True)

    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only,
        token_spans=eval(f"{token_spans}")
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))

    # --- Write structured analysis alongside the image ---
    # Build a detection list with parsed labels/scores and both normalized and pixel bboxes
    W, H = image_pil.size  # width, height
    detections = []
    chip_metadata = []

    for idx, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
        # `label` often looks like "a cat(0.36)"; parse out score if present
        m = re.search(r"\((0?\.?\d+(?:\.\d+)?)\)$", label)
        score = float(m.group(1)) if m else None
        phrase = re.sub(r"\(0?\.?\d+(?:\.\d+)?\)$", "", label).strip()

        cx, cy, w, h = [float(v) for v in box.tolist()]  # normalized cx,cy,w,h
        # convert to pixel xyxy
        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W)
        y1 = int((cy + h / 2) * H)

        detection_dict = {
            "label": phrase,
            "score": score,
            "bbox_norm": [cx, cy, w, h],  # normalized cx,cy,w,h (0..1)
            "bbox_xyxy": [x0, y0, x1, y1]  # pixel coordinates
        }

        # Extract chips if requested
        if args.extract_chips:
            chip, chip_bbox = extract_chip(image_pil, [x0, y0, x1, y1], args.chip_margin)

            # Save chip
            chip_filename = f"chip_{idx:03d}_{phrase.replace(' ', '_')}.jpg"
            chip_path = os.path.join(chips_dir, chip_filename)
            chip.save(chip_path, quality=95)  # High quality to preserve details

            chip_info = {
                "detection_idx": idx,
                "filename": chip_filename,
                "original_bbox": [x0, y0, x1, y1],
                "chip_bbox": chip_bbox,
                "margin_used": args.chip_margin,
                "chip_size": [chip.width, chip.height]
            }

            # Calculate quality metrics if requested
            if args.chip_quality:
                try:
                    from scipy import ndimage

                    quality_metrics = calculate_chip_quality_metrics(chip)
                    chip_info["quality_metrics"] = quality_metrics
                except ImportError:
                    print("Warning: scipy not installed, skipping quality metrics")

            chip_metadata.append(chip_info)
            detection_dict["chip_info"] = chip_info

        detections.append(detection_dict)

    # Create thumbnail with masks if requested
    if args.create_thumbnail:
        thumbnail = create_thumbnail_with_masks(image_pil, detections, args.thumbnail_size)
        thumbnail_path = os.path.join(output_dir, "thumbnail_with_masks.jpg")
        thumbnail.save(thumbnail_path)
        print(f"Saved thumbnail with masks to {thumbnail_path}")

    summary = {
        "image_path": image_path,
        "text_prompt": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "size": {"width": W, "height": H},
        "detections": detections,
    }

    # Add chip extraction info if chips were extracted
    if args.extract_chips:
        summary["chip_extraction"] = {
            "enabled": True,
            "margin_percent": args.chip_margin,
            "chips_directory": chips_dir,
            "total_chips": len(chip_metadata),
            "quality_metrics_enabled": args.chip_quality
        }

    # Save JSON file
    json_path = os.path.join(output_dir, "pred.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save separate chip metadata file if chips were extracted
    if args.extract_chips:
        chip_meta_path = os.path.join(chips_dir, "chip_metadata.json")
        with open(chip_meta_path, "w") as f:
            json.dump(chip_metadata, f, indent=2)
        print(f"Extracted {len(chip_metadata)} chips to {chips_dir}")

    # Also save a simple TSV for quick grepping/spreadsheets
    tsv_path = os.path.join(output_dir, "pred.tsv")
    with open(tsv_path, "w") as f:
        f.write("label\tscore\tcx\tcy\tw\th\tx0\ty0\tx1\ty1")
        if args.extract_chips:
            f.write("\tchip_file")
        f.write("\n")

        for i, d in enumerate(detections):
            s = "" if d["score"] is None else f"{d['score']:.4f}"
            cx, cy, w, h = d["bbox_norm"]
            x0, y0, x1, y1 = d["bbox_xyxy"]
            f.write(f"{d['label']}\t{s}\t{cx:.6f}\t{cy:.6f}\t{w:.6f}\t{h:.6f}\t{x0}\t{y0}\t{x1}\t{y1}")
            if args.extract_chips:
                f.write(f"\t{chip_metadata[i]['filename']}")
            f.write("\n")

    print(f"Detection results saved to {output_dir}")
    if args.extract_chips:
        print(f"Quality filtering suggestions:")
        if args.chip_quality and chip_metadata:
            # Analyze quality metrics across all chips
            blurry_count = sum(1 for c in chip_metadata if c.get('quality_metrics', {}).get('is_blurry', False))
            overexposed_count = sum(
                1 for c in chip_metadata if c.get('quality_metrics', {}).get('is_overexposed', False))
            underexposed_count = sum(
                1 for c in chip_metadata if c.get('quality_metrics', {}).get('is_underexposed', False))

            print(f"  - Blurry chips: {blurry_count}/{len(chip_metadata)}")
            print(f"  - Overexposed chips: {overexposed_count}/{len(chip_metadata)}")
            print(f"  - Underexposed chips: {underexposed_count}/{len(chip_metadata)}")