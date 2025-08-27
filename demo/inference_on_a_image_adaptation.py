import argparse
import os
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class RealEstateObjectDetector:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = device
        self.model = self.load_model(config_path, checkpoint_path)

    def load_model(self, config_path, checkpoint_path):
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def detect_interior_objects(self, image_path):
        """Detect common interior objects in real estate photos"""
        interior_prompts = [
            "sofa . couch . chair . dining table . coffee table",
            "television . fireplace . chandelier . ceiling fan",
            "kitchen island . refrigerator . stove . dishwasher",
            "bed . dresser . nightstand . closet",
            "bathtub . shower . toilet . sink . vanity"
        ]

        all_detections = []
        for prompt in interior_prompts:
            detections = self.detect(image_path, prompt,
                                     box_threshold=0.35,
                                     text_threshold=0.25)
            all_detections.extend(detections)
        return all_detections

    def detect_exterior_objects(self, image_path):
        """Detect common exterior objects in real estate photos"""
        exterior_prompts = [
            "garage . driveway . front door . porch . deck",
            "swimming pool . hot tub . patio . pergola",
            "fence . gate . mailbox . landscaping",
            "window . roof . siding . brick wall"
        ]

        all_detections = []
        for prompt in exterior_prompts:
            detections = self.detect(image_path, prompt,
                                     box_threshold=0.30,  # Lower threshold for exterior
                                     text_threshold=0.20)
            all_detections.extend(detections)
        return all_detections

    def detect(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        # Load and preprocess image
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)

        # Format text prompt
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        # Run inference
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # Filter results
        logits_filt = logits.cpu()
        boxes_filt = boxes.cpu()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # Get phrases
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        results = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            confidence = logit.max().item()

            # Convert box to pixel coordinates
            H, W = image_pil.size[::-1]
            box_pixels = box * torch.Tensor([W, H, W, H])
            box_pixels[:2] -= box_pixels[2:] / 2
            box_pixels[2:] += box_pixels[:2]

            results.append({
                'object': pred_phrase,
                'confidence': confidence,
                'bbox': box_pixels.tolist()
            })

        return results