from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from PIL import Image
import json, os, csv
import logging
from tqdm import tqdm
import torch

# Set the path to your test data directory
test_data_dir = "Data/test_data/test_data"

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("test_model/checkpoint-525")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")

# Create a list to store the results
results = []

# Iterate through each file in the test data directory
samples = os.listdir(test_data_dir)
for filename in tqdm(os.listdir(test_data_dir), desc="Processing"):
    sample_path = f"Data/test_data/{filename}"

    # Read the json file
    json_path = os.path.join(sample_path, "data.json")
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        question = data["question"]
        image_id = data["id"]

    # Read the corresponding image
    image_path = os.path.join(test_data_dir, f"{image_id}", "image.png")
    image = Image.open(image_path).convert("RGB")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

    out = model.generate(**encoding)
    generated_text = processor.decode(out[0], skip_special_tokens=True)


    results.append((image_id, generated_text))

# Write the results to a CSV file
csv_file_path = "Results/results.csv"
with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ID", "Label"])  # Write header
    csv_writer.writerows(results)

print(f"Results saved to {csv_file_path}")