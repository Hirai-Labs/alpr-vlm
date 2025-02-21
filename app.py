import ast
import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "Hirai-Labs/FT-SmolVLM-500M-Instruct-ALPR",
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are an AI assistant whose job is to inspect an image and provide the desired information from the image. If the desired field is not clear or not well detected, return None for this field. Do not try to guess."},
                {"type": "image"},
                {"type": "text", "text": 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"type": {"title": "Type", "description": "Return the type of the vehicle", "examples": ["Car", "Truck", "Motorcycle", "Bus"], "type": "string"}, "license_plate": {"title": "License Plate", "description": "Return the license plate number of the vehicle", "type": "string"}, "make": {"title": "Make", "description": "Return the Make of the vehicle", "examples": ["Toyota", "Honda", "Ford", "Suzuki"], "type": "string"}, "model": {"title": "Model", "description": "Return the model of the vehicle", "examples": ["Corolla", "Civic", "F-150"], "type": "string"}, "color": {"title": "Color", "description": "Return the color of the vehicle", "examples": ["Red", "Blue", "Black", "White"], "type": "string"}}, "required": ["type", "license_plate", "make", "model", "color"]}\n```'}
            ]
        }
    ]

def predictor(image):
    image = load_image(image=image)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    output = generated_texts[0]


    assistant_part = output.split("Assistant: ")[1]        
    dict_data = ast.literal_eval(assistant_part)
    return dict_data

iface = gr.Interface(
    fn=predictor,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=["images/image1.jpg", "images/image2.jpg", "images/image3.jpg", "images/image4.jpg"]
)

iface.launch(server_name="0.0.0.0", server_port=8080)