#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/stable_unclip
def stable_unclip_example() -> None:
	import torch
	import transformers
	import diffusers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		# Text-to-Image Generation

		from diffusers.models import PriorTransformer

		prior_model_id = "kakaobrain/karlo-v1-alpha"
		data_type = torch.float16
		prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

		prior_text_model_id = "openai/clip-vit-large-patch14"
		prior_tokenizer = transformers.CLIPTokenizer.from_pretrained(prior_text_model_id)
		prior_text_model = transformers.CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
		prior_scheduler = diffusers.UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
		prior_scheduler = diffusers.DDPMScheduler.from_config(prior_scheduler.config)

		stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

		pipe = diffusers.StableUnCLIPPipeline.from_pretrained(
			stable_unclip_model_id,
			torch_dtype=data_type,
			variant="fp16",
			prior_tokenizer=prior_tokenizer,
			prior_text_encoder=prior_text_model,
			prior=prior,
			prior_scheduler=prior_scheduler,
		)

		pipe = pipe.to("cuda")
		wave_prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

		image = pipe(prompt=wave_prompt).images[0]
		image.save("./image.png")

	if True:
		# Text guided Image-to-Image Variation

		from diffusers.utils import load_image

		pipe = diffusers.StableUnCLIPImg2ImgPipeline.from_pretrained(
			"stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
		)
		pipe = pipe.to("cuda")

		url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
		init_image = load_image(url)

		if True:
			image = pipe(init_image).images[0]
		else:
			prompt = "A fantasy landscape, trending on artstation"

			image = pipe(init_image, prompt=prompt).images[0]
		image.save("./variation_image.png")

	if True:
		#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

		if True:
			pipe = diffusers.StableUnCLIPPipeline.from_pretrained(
				"fusing/stable-unclip-2-1-l",
				torch_dtype=torch.float16,
			)  # TODO update model path
		elif False:
			pipe = diffusers.StableDiffusionPipeline.from_pretrained(
				"runwayml/stable-diffusion-v1-5",
				torch_dtype=torch.float16,
				use_safetensors=True,
			)
		else:
			pipe = diffusers.DiffusionPipeline.from_pretrained(
				"stabilityai/stable-diffusion-2-1",
				torch_dtype=torch.float16,
			)
		pipe = pipe.to("cuda")

		prompt = "a photo of an astronaut riding a horse on mars"
		#pipe.enable_attention_slicing()
		#pipe.enable_vae_slicing()
		#pipe.enable_vae_slicing()
		#pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
		# Workaround for not accepting attention shape using VAE for Flash Attention
		#pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
		images = pipe(prompt).images
		images[0].save("./astronaut_horse.png")

	if True:
		import requests
		from PIL import Image
		from io import BytesIO

		pipe = diffusers.StableUnCLIPImg2ImgPipeline.from_pretrained(
			"stabilityai/stable-diffusion-2-1-unclip-small",
			torch_dtype=torch.float16,
		)
		pipe = pipe.to("cuda")

		url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

		response = requests.get(url)
		init_image = Image.open(BytesIO(response.content)).convert("RGB")
		init_image = init_image.resize((768, 512))

		prompt = "A fantasy landscape, trending on artstation"

		images = pipe(init_image, prompt).images
		images[0].save("./fantasy_landscape.png")

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/unclip
def unclip_example() -> None:
	# No contents
	raise NotImplementedError

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/wuerstchen
def wuerstchen_example() -> None:
	import torch
	import diffusers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		# Text-to-Image Generation

		from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

		pipe = diffusers.AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to(device)

		caption = "Anthropomorphic cat dressed as a fire fighter"
		images = pipe(
			caption,
			width=1024,
			height=1536,
			prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
			prior_guidance_scale=4.0,
			num_images_per_prompt=2,
		).images

	if True:
		from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

		dtype = torch.float16
		num_images_per_prompt = 2

		prior_pipeline = diffusers.WuerstchenPriorPipeline.from_pretrained(
			"warp-ai/wuerstchen-prior", torch_dtype=dtype
		).to(device)
		decoder_pipeline = diffusers.WuerstchenDecoderPipeline.from_pretrained(
			"warp-ai/wuerstchen", torch_dtype=dtype
		).to(device)

		# Speed-Up Inference
		#prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
		#decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)

		caption = "Anthropomorphic cat dressed as a fire fighter"
		negative_prompt = ""

		prior_output = prior_pipeline(
			prompt=caption,
			height=1024,
			width=1536,
			timesteps=DEFAULT_STAGE_C_TIMESTEPS,
			negative_prompt=negative_prompt,
			guidance_scale=4.0,
			num_images_per_prompt=num_images_per_prompt,
		)
		decoder_output = decoder_pipeline(
			image_embeddings=prior_output.image_embeddings,
			prompt=caption,
			negative_prompt=negative_prompt,
			guidance_scale=0.0,
			output_type="pil",
		).images[0]
		print(decoder_output)

	if False:
		pipe = diffusers.WuerstchenCombinedPipeline.from_pretrained("warp-ai/Wuerstchen", torch_dtype=torch.float16).to(device)
		prompt = "an image of a shiba inu, donning a spacesuit and helmet"
		images = pipe(prompt=prompt)

	if False:
		prior_pipe = diffusers.WuerstchenPriorPipeline.from_pretrained(
			"warp-ai/wuerstchen-prior", torch_dtype=torch.float16
		).to(device)

		prompt = "an image of a shiba inu, donning a spacesuit and helmet"
		prior_output = pipe(prompt)

	if False:
		prior_pipe = diffusers.WuerstchenPriorPipeline.from_pretrained(
			"warp-ai/wuerstchen-prior", torch_dtype=torch.float16
		).to(device)
		gen_pipe = diffusers.WuerstchenDecoderPipeline.from_pretrain(
			"warp-ai/wuerstchen", torch_dtype=torch.float16
		).to(device)

		prompt = "an image of a shiba inu, donning a spacesuit and helmet"
		prior_output = pipe(prompt)
		images = gen_pipe(prior_output.image_embeddings, prompt=prompt)

# REF [site] >> https://huggingface.co/zai-org
def cog_view_example():
	# Models:
	#	zai-org/CogView3-Plus-3B
	#
	#	zai-org/CogView4-6B

	if False:
		import torch
		import diffusers
		
		pipe = diffusers.CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.float16).to("cuda")

		# Enable it to reduce GPU memory usage
		pipe.enable_model_cpu_offload()
		pipe.vae.enable_slicing()
		pipe.vae.enable_tiling()

		prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
		image = pipe(
			prompt=prompt,
			guidance_scale=7.0,
			num_images_per_prompt=1,
			num_inference_steps=50,
			width=1024,
			height=1024,
		).images[0]

		image.save("./cogview3.png")

	if True:
		import torch
		import diffusers

		pipe = diffusers.CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

		# Open it for reduce GPU memory usage
		pipe.enable_model_cpu_offload()
		pipe.vae.enable_slicing()
		pipe.vae.enable_tiling()

		prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
		image = pipe(
			prompt=prompt,
			guidance_scale=3.5,
			num_images_per_prompt=1,
			num_inference_steps=50,
			width=1024,
			height=1024,
		).images[0]

		image.save("./cogview4.png")

# REF [site] >> https://huggingface.co/Qwen
def qwen_image_example():
	# Models:
	#	Qwen/Qwen-Image
	#	Qwen/Qwen-Image-Edit
	#	Qwen/Qwen-Image-Edit-2509

	import torch
	import diffusers

	if True:
		model_name = "Qwen/Qwen-Image"

		# Load the pipeline
		if torch.cuda.is_available():
			torch_dtype = torch.bfloat16
			device = "cuda"
		else:
			torch_dtype = torch.float32
			device = "cpu"

		pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
		pipe = pipe.to(device)

		positive_magic = {
			"en": ", Ultra HD, 4K, cinematic composition.",  # For english prompt
			"zh": ", 超清，4K，电影级构图."  # For chinese prompt
		}

		# Generate image
		prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

		negative_prompt = " "  # Using an empty string if you do not have specific concept to remove

		# Generate with different aspect ratios
		aspect_ratios = {
			"1:1": (1328, 1328),
			"16:9": (1664, 928),
			"9:16": (928, 1664),
			"4:3": (1472, 1140),
			"3:4": (1140, 1472),
			"3:2": (1584, 1056),
			"2:3": (1056, 1584),
		}

		width, height = aspect_ratios["16:9"]

		image = pipe(
			prompt=prompt + positive_magic["en"],
			negative_prompt=negative_prompt,
			width=width,
			height=height,
			num_inference_steps=50,
			true_cfg_scale=4.0,
			generator=torch.Generator(device="cuda").manual_seed(42)
		).images[0]

		image.save("./example.png")

	if True:
		import os
		import torch
		import diffusers
		from PIL import Image

		pipeline = diffusers.QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
		print("pipeline loaded")
		pipeline.to(torch.bfloat16)
		pipeline.to("cuda")
		pipeline.set_progress_bar_config(disable=None)
		image = Image.open("./input.png").convert("RGB")
		prompt = "Change the rabbit's color to purple, with a flash light background."
		inputs = {
			"image": image,
			"prompt": prompt,
			"generator": torch.manual_seed(0),
			"true_cfg_scale": 4.0,
			"negative_prompt": " ",
			"num_inference_steps": 50,
		}

		with torch.inference_mode():
			output = pipeline(**inputs)
			output_image = output.images[0]
			output_image.save("./output_image_edit.png")
			print("image saved at", os.path.abspath("output_image_edit.png"))

	if True:
		import os
		import torch
		import diffusers
		from PIL import Image

		pipeline = diffusers.QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
		print("pipeline loaded")

		pipeline.to("cuda")
		pipeline.set_progress_bar_config(disable=None)
		image1 = Image.open("./input1.png")
		image2 = Image.open("./input2.png")
		prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
		inputs = {
			"image": [image1, image2],
			"prompt": prompt,
			"generator": torch.manual_seed(0),
			"true_cfg_scale": 4.0,
			"negative_prompt": " ",
			"num_inference_steps": 40,
			"guidance_scale": 1.0,
			"num_images_per_prompt": 1,
		}

		with torch.inference_mode():
			output = pipeline(**inputs)
			output_image = output.images[0]
			output_image.save("./output_image_edit_plus.png")
			print("image saved at", os.path.abspath("output_image_edit_plus.png"))

def main():
	# Diffusion models
	#	Refer to ./diffusion_model_test.py

	# unCLIP
	#stable_unclip_example()
	#unclip_example()  # No contents

	#wuerstchen_example()

	#-----
	# Text-to-image generation

	#cog_view_example()  # CogView3, CogView4
	qwen_image_example()  # Qwen-Image

	#-----
	# Vision and language.
	#	Refer to language_model_test.py
#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
