import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import random
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
from colorama import Fore
import re

model_dir = '/home/jxliu/models/Qwen2-VL-7B-Instruct/'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def score_annotation(image_paths, annotation: dict):
    flattened_data = []
    try:
        for key, value in annotation.items():
            if isinstance(value, list):
                if key != "All parts":
                    flattened_data.append(f"\"{key}\": \"{value}\"")
                    continue
                flattened_data.append(f"\"All parts\":")
                for idx, part in enumerate(value):
                    # print(part)
                    flattened_data.append(f"\"Part {idx + 1} item\": \"{part['Part item']}\"")
                    flattened_data.append(f"\"Part {idx + 1} style\": \"{part['Part style']}\"")
                    flattened_data.append(f"\"Part {idx + 1} colors\": \"{part['Part colors']}\"")
                    flattened_data.append(f"\"Part {idx + 1} shape\": \"{part['Part shape']}\"")
                    flattened_data.append(f"\"Part {idx + 1} size\": \"{part['Part size']}\"")
                    flattened_data.append(f"\"Part {idx + 1} position\": \"{part['Part position']}\"")
                    flattened_data.append(f"\"Part {idx + 1} relationship to entirety\": \"{part['Part relationship to entirety']}\"")
                    flattened_data.append(f"\"Part {idx + 1} extra description\": \"{part['Part extra description']}\"\n")
            else:
                flattened_data.append(f"\"{key}\": \"{value}\"")
    except KeyError as e:
        print(Fore.RED + f"Error occurs: {e}")
        return None

    # Join all the lines into a single string
    result_string = "\n".join(flattened_data)
    instruction = f"""
    Read the description of an image below, and choose an image which matches the description most from the given images. ONLY return the index of the chosen image(start from 1). Any other information is innecessary.
    # Description #:\n 
    {result_string}
    \nPlease note that there are some explanations below about the description: \n
    1. "null" means no suitable description.\n
    2. Key "All parts" contains detailed information about the various parts that make up the entirety.\n 
    3. Key "Part size" is the ratio of the area occupied by this part to the total area.\n
    4. Key "Part position" is the bounding box coordinates of this part in the format (xl, y1, x2, y2), where the coordinates are normalized between 0 and 1.\n
    Now the images are given as below. Choose one and return its index.
    """
    # print(Fore.BLUE + instruction)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}] + [
                {"type": "image", "image": path} for path in image_paths
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(Fore.YELLOW + output_text[0])

    index = re.findall(r'\d+', output_text[0])
    if index:
        return int(index[0]) 
    else: 
        return None

def get_images(all_images, target_image_path, num_samples=1):
    selected_images = []
    while len(selected_images) < num_samples:
        new_image = random.choice(all_images)
        if not any(word in os.path.basename(new_image).split('-') for word in os.path.basename(target_image_path).split('-')):
            selected_images.append(new_image)
    return selected_images

def process_annotation_and_image(annotation_file, image_folder, resume_file, output_folder, all_images):
    results = []
    processed_files = []
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            processed_files = json.load(f)
    
    with open(annotation_file, 'r') as f:
        for line in tqdm(f, desc='Processing annotations', unit='annotation'):
            line = line.replace("desciption", "description")
            annotations = json.loads(line)
            if not annotations['feature']:
                continue


            svg_name = annotations['svg'].replace('.svg', '')
            if svg_name not in processed_files:
                print(Fore.BLUE + svg_name)
                annotation_words = svg_name.split('-')
                # annotation_text = json.dumps(annotations['feature'])
                    
                json_filename = os.path.splitext(os.path.basename(annotation_file))[0]
                target_image_path = os.path.join(image_folder, json_filename, f"{svg_name}.png")
                    
                if not os.path.exists(target_image_path):
                    print(f"Target image not found for {svg_name} in {json_filename}")
                    continue
  
                scores = 0
                for _ in range(10):
                    selected_images = get_images(all_images, target_image_path, num_samples=9) + [target_image_path]
                    random.shuffle(selected_images)  
                    best_match_index = score_annotation(selected_images, annotations["feature"])
                    if not best_match_index:
                        continue
                    target_index = selected_images.index(target_image_path) + 1
                        
                    print(f"SVG: {svg_name}, Round {_+1}, Images: {[selected_image.split('/')[-1] for selected_image in selected_images]}, Best Match Index: {best_match_index}, Target Position: {target_index}")
                        
                    if best_match_index == target_index:
                        scores += 1

                annotations['score'] = scores  
                output_file_path = os.path.join(output_folder, f"{json_filename}.json")
                with open(output_file_path, 'a') as out_f:
                    json.dump(annotations, out_f)
                    out_f.write('\n')
                    
                results.append({
                    'svg': svg_name,
                    'score': scores
                })
                print(f"score: {scores}")

                annotations['score'] = scores  
                processed_files.append(svg_name)

                with open(resume_file, 'w') as f:
                    json.dump(processed_files, f)            
    return results

pizhu_folder_path = './annotation'
image_folder_path = './small_png'
resume_file_path = './has_scored.json'
output_folder_path = './annotation_score'

all_images = []
for root, dirs, files in os.walk(image_folder_path):
    for file in files:
        #if file.endswith('.png') and not any(word in file.split('-') for word in annotation_words):
        all_images.append(os.path.join(root, file))

results = []
for annotation_file_name in os.listdir(pizhu_folder_path):
    if annotation_file_name.endswith('.json'):
        print(f"Processing {annotation_file_name}")
        annotation_file_path = os.path.join(pizhu_folder_path, annotation_file_name)
        file_results = process_annotation_and_image(annotation_file_path, image_folder_path, resume_file_path, output_folder_path, all_images)
        results.extend(file_results)

# for result in results:
#     print(f"SVG: {result['svg']}, Score: {result['score']}")