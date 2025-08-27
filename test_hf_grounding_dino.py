# test_hf_grounding_dino.py - GPU version fixed
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import cv2

# Use base model
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model device: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")

image = Image.open(".asset/cat_dog.jpeg")
text = "one cat . one dog ."

inputs = processor(images=image, text=text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs["input_ids"],  # Fixed: dictionary access
    target_sizes=[image.size[::-1]]
)

box_threshold = 0.35
boxes = results[0]["boxes"]
scores = results[0]["scores"]
labels = results[0]["labels"]

# Filter and save annotated image
img_cv2 = cv2.imread(".asset/cat_dog.jpeg")
detected_count = 0

for i, score in enumerate(scores):
    if score > box_threshold:
        detected_count += 1
        x1, y1, x2, y2 = [int(v.item()) for v in boxes[i]]

        color = (0, 255, 0) if "cat" in labels[i] else (255, 0, 0)

        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv2, f"{labels[i]} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imwrite("output_both_detected.jpg", img_cv2)
print(f"Detected {detected_count} objects. Saved to output_both_detected.jpg")