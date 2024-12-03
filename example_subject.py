import torch, os
from diffusers.pipelines import FluxPipeline
from src.condition import Condition
from PIL import Image

from transformers import CLIPTextModel

from src.generate import generate, seed_everything

def test_omini_control(model_path):
    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    print("Pipeline loaded")
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name=f"omini/subject_512.safetensors",
        adapter_name="subject",
    )
    image = Image.open("assets/penguin.jpg").convert("RGB").resize((512, 512))

    condition = Condition("subject", image)

    prompt = "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat."

    seed_everything(0)

    print("Generating image...")
    result_img = generate(
        pipe,
        prompt=prompt,
        conditions=[condition],
        num_inference_steps=8,
        height=512,
        width=512,
    ).images[0]

    concat_image = Image.new("RGB", (1024, 512))
    concat_image.paste(image, (0, 0))
    concat_image.paste(result_img, (512, 0))

    # Save the image to disk:
    concat_image.save("output.jpg")

    print("done!")