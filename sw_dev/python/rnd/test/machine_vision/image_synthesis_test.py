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

def main():
	# Diffusion models
	#	Refer to ./diffusion_model_test.py

	# unCLIP
	stable_unclip_example()
	#unclip_example()  # No contents

	wuerstchen_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
