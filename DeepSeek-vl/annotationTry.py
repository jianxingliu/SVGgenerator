import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import AutoModelForCausalLM
import json
from colorama import Fore
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the path to the model
model_path = "/home/jxliu/models/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

instruction = """
Review the image given, please give the text prompt that could generate a svg file whose rendered result is this image. Note that give the text prompt according to the following json format.\n
```json
{"Overall name":<answer>, "Overall event": <answer>, "Overall style": <answer>, "Overall colors": <answer>, "Overallshape": <answer>, 
"All parts": [{"Part item": <answer>, "Part style": <answer>, "Part colors": <answer>,"Part shape": <answer>, "Part size": <answer>, "Part position": <answer>, "Part relationship to entirety": <answer>, "Part extra desciption": <answer>}],
"Overall extra description". <answer>}
```
\n\nPlease note that there are some explanations below on how you should answer: \n
1. You should replace <answer> with your answer, lf you don't have a suitable answer, you can answer "null".\n
2. For key "Overall name", please give the entirety a name in a few short words.\n 
3. For key "Overall event", please briefly describe the event oraction represented by the entirety in one sentence.\n
4, For key "Overall style", please answer overall style in a few short words.\n 
5. For key "Overall colors", please answer in a few short words all the colors used for the entirety.\n
6. For key "Overall shape", if the entirety has clear semantic information, please answer "null" here. lf the entirety does not have clear semantic information, please give a common shape noun to describe the shape of the entirety. Note that shape nouns are selected from the following list: Circle, Square, Rectangle, Triangle, Ellipse, Polygon, Parallelogram, Rhombus, Trapezoid, Star, Heart, Crescent, Diamond, Semicircle, Arrow.\n
7. For key "All parts", its value is a list of dictionaries containing detailed information about the various parts that make up the entirety.\n 
8. For key "Part item" in one of values of key "All parts", please use a few short words to answer what this part represents, the thing, event or action.\n
9. For key "'Part style" in one of values of key "All parts", please answer this part's style in a fewshort words.\n 
10. For key "Part colors" in one of values of key "All parts", please answer in a few short wordsall the colors used for this part.\n
11. For key "Part shape" in one of values of key "All parts", please use a common shape noun to describe the shape of this part. Note that shape nouns are selected from the following list: Circle, Square, Rectangle, Triangle, Ellipse, Polygon, Parallelogram, Rhombus, Trapezoid,Star, Heart, Crescent, Diamond, Semicircle, Arrow.\n
12. For key "Part size" in one of values of key "All parts", please answer the ratio of the area occupied by this part to the total area and answer as a percentage.Note that round the number before the percentage sign to an integer.\n
13. For key "Part position" in one of values of key "All parts", please provide the bounding box coordinates of this part in the format (xl, y1, x2, y2), where the coordinates are normalized between 0 and 1.\n
14. For key "Part relationship to entirety" in one of values of key "All parts", please use one sentence to answer the position and function of this partrelative to the entirety. lf there is no other valid information, you can answer "null".\n
15. For key "Part extradesciption" in one of values of key "All parts", please use a few sentences to add other valid information of this part besides the existing key value in one of values of key "All parts". if there is no other valid information, you can answer "null".\n
16. For key "Overall extra desciption", please use a few sentences to add other valid information of this part besides the existing key value. lf there is no other validinformation, you can answer "null".\n
Describe the image in a neutral tone, focusing only on the objects and their characteristics without any emotions or metaphors.\n
"""
# The image is given below.\n
# Below is an example of the json format of the text prompt for the image:\n
example = {
    "Overall name":"Penguin Icon",
    "Overall event":"A stylized representation of a penguin.",
    "Overall style":"Minimalist line art",
    "Overall colors":"Black",
    "Overall shape":"Rectangle",
    "All parts": [
        {
            "Part item":"Head",
            "Part style":"Minimalist",
            "Part colors":"Black",
            "Part shape":"Rectangle",
            "Part size":"20%",
            "Part position":"Top center",
            "Part relationship to entirety": "The head is the upper part of the penguin, containing the eyes andbeak.",
            "Part extra description": "The head is the most detailed part of the penguin, with two eyes and a beak.",
        },
        {
            "Part item":"Body",
            "Part style":"Minimalist",
            "part colors":"Black",
            "Part shape":"Rectangle",
            "Part size":"60%",
            "Part position":"center",
            "Part relationship to entirety":"The body is the main part of the penguin, extending downwards from thehead.",
            "Part extra description": "The body is the largest part of the penguin, representing its torsoand legs.",
        },
        {
            "Part item": "Flippers",
            "Part style":"Minimalist",
            "Part colors":"Black",
            "Part shape":"Rectangle",
            "Part size":"10%",
            "Part position":"Bottom",
            "Part relationship to entirety":"The flippers are the lower part of the penguin, extending from thebody.",
            "Part extra description":"The flippers are the penguin's feet, used for swimming.",
        }
    ],
    "Overall extra description":"This minimalist line art penguin icon is simple yet recognizable, using only black lines to depict the penguin's head, body, and flippers."
}
example_txt = """Below is an example of the json format of the text prompt for the image:\n
                # Image # :\n <image_placeholder> 
                # Text Prompt #:\n"""
example_txt += "```json" + json.dumps(example) + "```\n"
example_txt += "Use the example above as a format reference, but do not copy it. Focus on describing the image in a neutral, factual manner.\n"
query = instruction + example_txt + """The image you need to generate the text prompt for is as below.\n <image_placeholder> 
Now please generate your result in a strict JSON format:"""
png_dir = "../rgb_png"

for file in os.listdir("../rgb_png"):
    conversation = [
        # {
        #     "role": "User",
        #     "content": instruction,
        #     "images": ["./example800.png"]
        # },
        # {
        #     "role": "Assistant",
        #     "content": example_txt
        # },
        {
            "role": "User",
            "content": query,
            "images": [os.path.join(png_dir, file)]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

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
    print(Fore.BLUE + file)
    print(Fore.YELLOW + answer)
    try:
        answer = json.loads(answer)
        print(Fore.GREEN + answer["Overall extra description"])
    except:
        print(Fore.RED + "Invaild!")