from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

device = "cpu"

proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B").to(device)

img = Image.open("IMG_1530.JPG")
inputs = proc(text="What is shown here? Describe the image and the mood of the person if there is/are any.", images=[img], return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10)

print(proc.batch_decode(out))
