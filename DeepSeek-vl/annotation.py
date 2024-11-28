import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from transformers import AutoModelForCausalLM
import json
from colorama import Fore
from tqdm import tqdm
import defusedxml

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from readSVG import svg2rgb
from instrction import get_conversation


# specify the path to the model
model_path = "/home/jxliu/models/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


# The image is given below.\n
# Below is an example of the json format of the text prompt for the image:\n

# example_txt = "```json" + json.dumps(example) + "```\n"
# query = instruction + example_txt + """The image you need to generate the text prompt for is as below.\n <image_placeholder> 
# Now please generate your result in a strict JSON format:"""
png_dir = "../rgb_png"
svg_dir = "../svg/vector"
result_dir = "../annotation"
all_file_num = 0

for folder in os.listdir(svg_dir):
    result_json = folder + '.json'
    result_path = os.path.join(result_dir, result_json)
    folder_dir = os.path.join(svg_dir, folder)
    num_file = len(os.listdir(folder_dir))
    all_file_num += num_file
    pbar = tqdm(os.listdir(folder_dir))

    for file in pbar:
        pbar.set_description(folder)
        
        png_path = os.path.join(png_dir, folder, file.split('.')[0] + '.png')
        if os.path.exists(png_path):
            continue
        print(Fore.BLUE + file)
        try:
            svg2rgb(folder, file)
        except Exception as e:
            print(Fore.RED + 'Error occurs!!')
            print(e)
            continue
        conversation = get_conversation(mode="0-shot", png_path=png_path)

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            # do_sample=True,
            # temperature=0.01
        )

        answer = tokenizer.decode(outputs[0].cuda().tolist(), skip_special_tokens=True)
        if "```json" in answer:
            answer = answer.split("```json")[1].split('```')[0]
        print(Fore.YELLOW + answer)

        try:
            answer = json.loads(answer)
            with open(result_path, 'a') as result_file:
                result = {"svg": file, "feature": answer}
                result = json.dumps(result)
                result_file.write(result + '\n')
            # print(Fore.GREEN + answer["Overall extra description"])
        except:
            with open(result_path, 'a') as result_file:
                result = {"svg": file, "feature": ""}
                result = json.dumps(result)
                result_file.write(result + '\n')
            print(Fore.RED + "Invalid answer!")
    
    print(Fore.GREEN + f"Already annotate {all_file_num} svg files")
        


