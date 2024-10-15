#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import PIL.Image
import tqdm

# REF [site] >> https://github.com/huggingface/diffusers
def diffusers_quickstart():
	from io import BytesIO
	import requests
	import torch
	import diffusers

	device = "cuda"

	# Text-to-Image generation with Stable Diffusion.
	#	pip install --upgrade diffusers transformers accelerate
	if True:
		# We recommend using the model in half-precision (fp16) as it gives almost always the same results as full precision while being roughly twice as fast and requiring half the amount of GPU RAM.
		#	REF [site] >> https://huggingface.co/runwayml/stable-diffusion-v1-5
		pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
	else:
		# Running the model locally.
		#	git lfs install
		#	git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
		pipe = diffusers.StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
	pipe = pipe.to(device)

	prompt = "a photo of an astronaut riding a horse on mars"
	image = pipe(prompt).images[0]

	image.show()
	#image.save("./astronaut_rides_horse_1.png")

	#----
	# If you are limited by GPU memory, you might want to consider chunking the attention computation in addition to using fp16.
	# The following snippet should result in less than 4GB VRAM.
	pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
	pipe = pipe.to(device)

	prompt = "a photo of an astronaut riding a horse on mars"
	pipe.enable_attention_slicing()
	image = pipe(prompt).images[0]

	image.show()
	#image.save("./astronaut_rides_horse_2.png")

	#----
	# If you wish to use a different scheduler (e.g.: DDIM, LMS, PNDM/PLMS), you can instantiate it before the pipeline and pass it to from_pretrained.
	pipe.scheduler = diffusers.LMSDiscreteScheduler.from_config(pipe.scheduler.config)

	prompt = "a photo of an astronaut riding a horse on mars"
	image = pipe(prompt).images[0]

	image.show()
	#image.save("./astronaut_rides_horse_lms.png")

	#----
	# If you want to run Stable Diffusion on CPU or you want to have maximum precision on GPU, please run the model in the default full-precision setting:
	pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
	# Disable the following line if you run on CPU.
	pipe = pipe.to(device)

	prompt = "a photo of an astronaut riding a horse on mars"
	image = pipe(prompt).images[0]

	image.show()
	#image.save("./astronaut_rides_horse_full_precision.png")

	#--------------------
	# JAX/Flax.
	#	Diffusers offers a JAX / Flax implementation of Stable Diffusion for very fast inference.
	#	JAX shines specially on TPU hardware because each TPU server has 8 accelerators working in parallel, but it runs great on GPUs too.

	#--------------------
	# Image-to-Image text-guided generation with Stable Diffusion.

	# Load the pipeline.
	if True:
		model_id_or_path = "runwayml/stable-diffusion-v1-5"
	else:
		# git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
		model_id_or_path = "./stable-diffusion-v1-5"
	pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
	pipe = pipe.to(device)

	# Let's download an initial image.
	url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

	response = requests.get(url)
	init_image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
	init_image = init_image.resize((768, 512))

	prompt = "A fantasy landscape, trending on artstation"
	images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

	images[0].show()
	#images[0].save("./fantasy_landscape.png")

	#--------------------
	# In-painting using Stable Diffusion.
	def download_image(url):
		response = requests.get(url)
		return PIL.Image.open(BytesIO(response.content)).convert("RGB")

	img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
	mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

	init_image = download_image(img_url).resize((512, 512))
	mask_image = download_image(mask_url).resize((512, 512))

	pipe = diffusers.StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
	pipe = pipe.to(device)

	prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
	image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

	image.show()

	#--------------------
	# Text-to-Image Latent Diffusion.
	#	pip install diffusers["torch"] transformers
	model_id = "CompVis/ldm-text2im-large-256"

	# Load model and scheduler.
	ldm = diffusers.DiffusionPipeline.from_pretrained(model_id)
	ldm = ldm.to(device)

	# Run pipeline in inference (sample random noise and denoise).
	prompt = "A painting of a squirrel eating a burger"
	image = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images[0]

	image.show()
	# Save image.
	#image.save("./squirrel.png")

	#--------------------
	# Unconditional Diffusion with discrete scheduler.
	#	pip install diffusers["torch"]
	model_id = "google/ddpm-celebahq-256"

	# Load model and scheduler.
	ddpm = diffusers.DDPMPipeline.from_pretrained(model_id)  # You can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference.
	#ddim = diffusers.DDIMPipeline.from_pretrained(model_id)
	#pndm = diffusers.PNDMPipeline.from_pretrained(model_id)
	ddpm.to(device)

	# Run pipeline in inference (sample random noise and denoise).
	image = ddpm().images[0]

	image.show()
	# Save image.
	#image.save("./ddpm_generated_image.png")

# REF [site] >> https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb
def diffusers_intro():
	import torch
	import diffusers

	image_pipe = diffusers.DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
	image_pipe.to("cuda")

	print(image_pipe)

	images = image_pipe().images
	images[0].show()

	#--------------------
	# Models.

	repo_id = "google/ddpm-church-256"
	model = diffusers.UNet2DModel.from_pretrained(repo_id)

	print("Model:")
	print(model)

	print("Model config:")
	print(model.config)

	model_random = diffusers.UNet2DModel(**model.config)
	model_random.save_pretrained("./my_model")

	model_random = diffusers.UNet2DModel.from_pretrained("./my_model")

	#--------------------
	# Inference.

	torch.manual_seed(0)

	noisy_sample = torch.randn(
		1, model.config.in_channels, model.config.sample_size, model.config.sample_size
	)
	print(f"noisy_sample.shape = {noisy_sample.shape}.")

	with torch.no_grad():
		noisy_residual = model(sample=noisy_sample, timestep=2).sample

	print(f"noisy_residual.shape = {noisy_residual.shape}.")

	#--------------------
	# Scheduler.

	scheduler = diffusers.DDPMScheduler.from_config(repo_id)
	print("Scheduler config:")
	print(scheduler.config)

	scheduler.save_config("my_scheduler")
	new_scheduler = diffusers.DDPMScheduler.from_config("my_scheduler")

	less_noisy_sample = scheduler.step(
		model_output=noisy_residual, timestep=2, sample=noisy_sample
	).prev_sample
	print(f"less_noisy_sample.shape = {less_noisy_sample.shape}.")

	#-----
	def display_sample(sample, i):
		image_processed = sample.cpu().permute(0, 2, 3, 1)
		image_processed = (image_processed + 1.0) * 127.5
		image_processed = image_processed.numpy().astype(np.uint8)

		image_pil = PIL.Image.fromarray(image_processed[0])
		display(f"Image at step {i}")
		display(image_pil)

	model.to("cuda")
	noisy_sample = noisy_sample.to("cuda")

	sample = noisy_sample
	for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
		# 1. predict noise residual.
		with torch.no_grad():
			residual = model(sample, t).sample

		# 2. compute less noisy image and set x_t -> x_t - 1.
		sample = scheduler.step(residual, t, sample).prev_sample

		# 3. optionally look at image.
		if (i + 1) % 50 == 0:
			display_sample(sample, i + 1)

	#-----
	scheduler = diffusers.DDIMScheduler.from_config(repo_id)
	scheduler.set_timesteps(num_inference_steps=50)

	sample = noisy_sample
	for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
		# 1. predict noise residual.
		with torch.no_grad():
			residual = model(sample, t).sample

		# 2. compute previous image and set x_t -> x_t - 1.
		sample = scheduler.step(residual, t, sample).prev_sample

		# 3. optionally look at image.
		if (i + 1) % 10 == 0:
			display_sample(sample, i + 1)

# REF [site] >> https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb
def diffusers_training_example():
	raise NotImplementedError

# REF [site] >>
#	https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
#	https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
#	https://huggingface.co/docs/diffusers/training/dreambooth
def diffusers_dreambooth_training_example():
	raise NotImplementedError

# REF [site] >> https://huggingface.co/CompVis
def compvis_example():
	# Models:
	#	CompVis/stable-diffusion-v1-1.
	#	CompVis/stable-diffusion-v1-2.
	#	CompVis/stable-diffusion-v1-3.
	#	CompVis/stable-diffusion-v1-4.

	import torch
	import diffusers

	# Install.
	#	pip install diffusers transformers scipy

	model_id = "CompVis/stable-diffusion-v1-4"
	device = "cuda"

	pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
	pipe = pipe.to(device)

	prompt = "a photo of an astronaut riding a horse on mars"
	image = pipe(prompt).images[0]
		
	image.save("./astronaut_rides_horse.png")

# REF [site] >> https://huggingface.co/stabilityai
def stabilityai_example():
	# Models:
	#	stabilityai/stable-diffusion-2.
	#	stabilityai/stable-diffusion-2-base.
	#	stabilityai/stable-diffusion-2-depth.
	#	stabilityai/stable-diffusion-2-inpainting.
	#	stabilityai/stable-diffusion-2-1.
	#	stabilityai/stable-diffusion-2-1-base.
	#	stabilityai/stable-diffusion-2-1-unclip.
	#	stabilityai/stable-diffusion-2-1-unclip-small.
	#	stabilityai/stable-diffusion-xl-base-0.9.
	#	stabilityai/stable-diffusion-xl-refiner-0.9.
	#	stabilityai/stable-diffusion-xl-base-1.0.
	#	stabilityai/stable-diffusion-xl-refiner-1.0.
	#	stabilityai/stable-diffusion-xl-1.0-tensorrt.
	#	stabilityai/sd-vae.
	#	stabilityai/sd-x2-latent-upscaler.
	#	stabilityai/sd-turbo.
	#	stabilityai/sdxl-vae.
	#	stabilityai/sdxl-turbo.
	#	stabilityai/sdxl-turbo-tensorrt.
	#	stabilityai/stable-video-diffusion-img2vid-xt-1-1.

	# REF [document] >>
	#	https://huggingface.co/docs/diffusers/using-diffusers/sdxl
	#	https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo

	import torch
	import diffusers

	# Install.
	#	pip install diffusers transformers accelerate scipy safetensors invisible_watermark

	if False:
		#model_id = "stabilityai/stable-diffusion-2"
		model_id = "stabilityai/stable-diffusion-2-base"

		# Use the Euler scheduler here instead.
		scheduler = diffusers.EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
		pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
		pipe = pipe.to("cuda")

		prompt = "a photo of an astronaut riding a horse on mars"
		image = pipe(prompt).images[0]
			
		image.save("./astronaut_rides_horse.png")

	if True:
		import requests

		pipe = diffusers.StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", torch_dtype=torch.float16).to("cuda")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		init_image = PIL.Image.open(requests.get(url, stream=True).raw)

		prompt = "two tigers"
		n_propmt = "bad, deformed, ugly, bad anotomy"
		image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]

	if True:
		pipe = diffusers.StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
		pipe.to("cuda")

		prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
		# image and mask_image should be PIL images.
		# The mask structure is white for inpainting and black for keeping as is
		image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

		image.save("./yellow_cat_on_park_bench.png")

	if True:
		model_id = "stabilityai/stable-diffusion-2-1"

		# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead.
		pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
		pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
		pipe = pipe.to("cuda")

		prompt = "a photo of an astronaut riding a horse on mars"
		image = pipe(prompt).images[0]

		image.save("./astronaut_rides_horse.png")

	if False:
		pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
		pipe.to("cuda")
		# If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda").
		#pipe.enable_model_cpu_offload()

		# If using torch < 2.0.
		#pipe.enable_xformers_memory_efficient_attention()

		# When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
		# Simple wrap the unet with torch compile before running the pipeline.
		#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

		prompt = "An astronaut riding a green horse"

		images = pipe(prompt=prompt).images[0]

	if False:
		pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
		pipe.to("cuda")
		# If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda").
		#pipe.enable_model_cpu_offload()

		# If using torch < 2.0.
		#pipe.enable_xformers_memory_efficient_attention()

		prompt = "An astronaut riding a green horse"

		image = pipe(prompt=prompt, output_type="latent").images

		pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
		pipe.to("cuda")
		# If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda").
		#pipe.enable_model_cpu_offload()

		# If using torch < 2.0.
		#pipe.enable_xformers_memory_efficient_attention()

		# When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
		# Simple wrap the unet with torch compile before running the pipeline.
		#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

		images = pipe(prompt=prompt, image=image).images

	if True:
		pipe = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
		pipe.to("cuda")
		# If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda").
		#pipe.enable_model_cpu_offload()

		# If using torch < 2.0.
		#pipe.enable_xformers_memory_efficient_attention()

		# When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
		# Simple wrap the unet with torch compile before running the pipeline.
		#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

		prompt = "An astronaut riding a green horse"

		images = pipe(prompt=prompt).images[0]

	if True:
		pipe = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
		pipe = pipe.to("cuda")
		# If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda").
		#pipe.enable_model_cpu_offload()

		# When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
		# Simple wrap the unet with torch compile before running the pipeline
		#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

		url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
		init_image = diffusers.utils.load_image(url).convert("RGB")
		prompt = "a photo of an astronaut riding a horse on mars"

		image = pipe(prompt, image=init_image).images

	if True:
		# Load both base & refiner.
		base = diffusers.DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
		base.to("cuda")
		refiner = diffusers.DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-xl-refiner-1.0",
			text_encoder_2=base.text_encoder_2,
			vae=base.vae,
			torch_dtype=torch.float16,
			use_safetensors=True,
			variant="fp16",
		)
		refiner.to("cuda")

		# Define how many steps and what % of steps to be run on each experts (80/20) here.
		n_steps = 40
		high_noise_frac = 0.8

		prompt = "A majestic lion jumping from a big stone at night"

		# Run both experts.
		image = base(
			prompt=prompt,
			num_inference_steps=n_steps,
			denoising_end=high_noise_frac,
			output_type="latent",
		).images
		image = refiner(
			prompt=prompt,
			num_inference_steps=n_steps,
			denoising_start=high_noise_frac,
			image=image,
		).images[0]

	if False:
		model = "CompVis/stable-diffusion-v1-4"
		vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
		#vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
		pipe = diffusers.StableDiffusionPipeline.from_pretrained(model, vae=vae)

	if False:
		pipeline = diffusers.StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
		pipeline.to("cuda")

		upscaler = diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
		upscaler.to("cuda")

		prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
		generator = torch.manual_seed(33)

		# We stay in latent space! Let's make sure that Stable Diffusion returns the image in latent space.
		low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

		upscaled_image = upscaler(
			prompt=prompt,
			image=low_res_latents,
			num_inference_steps=20,
			guidance_scale=0,
			generator=generator,
		).images[0]

		# Let's save the upscaled image under "upscaled_astronaut.png".
		upscaled_image.save("./astronaut_1024.png")

		# As a comparison: Let's also save the low-res image.
		with torch.no_grad():
			image = pipeline.decode_latents(low_res_latents)
		image = pipeline.numpy_to_pil(image)[0]

		image.save("./astronaut_512.png")

	if False:
		# Text-to-image.

		pipe = diffusers.AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
		pipe.to("cuda")

		prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
		image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

		#-----
		# Image-to-image.

		pipe = diffusers.AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
		pipe.to("cuda")

		init_image = diffusers.utils.load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))
		prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

		image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]

	if False:
		model = "stabilityai/your-stable-diffusion-model"
		vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
		pipe = diffusers.StableDiffusionPipeline.from_pretrained(model, vae=vae)

	if True:
		# Text-to-image.

		pipe = diffusers.AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
		pipe.to("cuda")

		prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

		image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

		#-----
		# Image-to-image.

		pipe = diffusers.AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
		pipe.to("cuda")

		init_image = diffusers.utils.load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))

		prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

		image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/unidiffuser
def uni_diffuser_example() -> None:
	import torch
	import diffusers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		# Unconditional Image and Text Generation

		model_id_or_path = "thu-ml/unidiffuser-v1"

		pipe = diffusers.UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
		pipe.to(device)

		if True:
			# Unconditional image and text generation. The generation task is automatically inferred
			#pipe.set_joint_mode()
			sample = pipe(num_inference_steps=20, guidance_scale=8.0)
			image = sample.images[0]
			text = sample.text[0]
		elif True:
			# Image-only generation
			pipe.set_image_mode()
			image = pipe(num_inference_steps=20).images[0]
			text = None
		elif False:
			# Text-only generation
			pipe.set_text_mode()
			text = pipe(num_inference_steps=20).text[0]
			image = None

		if image is not None: image.save("./unidiffuser_joint_sample_image.png")
		if text is not None: print(text)

	if True:
		# Text-to-Image Generation

		model_id_or_path = "thu-ml/unidiffuser-v1"

		pipe = diffusers.UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
		pipe.to(device)

		# Text-to-image generation
		prompt = "an elephant under the sea"

		sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
		t2i_image = sample.images[0]
		t2i_image

	if True:
		# Image-to-Text Generation

		from diffusers.utils import load_image

		model_id_or_path = "thu-ml/unidiffuser-v1"
		pipe = diffusers.UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
		pipe.to(device)

		# Image-to-text generation
		image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
		init_image = load_image(image_url).resize((512, 512))

		sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
		i2t_text = sample.text[0]
		print(i2t_text)

	if True:
		# Image Variation

		from diffusers.utils import load_image

		model_id_or_path = "thu-ml/unidiffuser-v1"

		pipe = diffusers.UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
		pipe.to(device)

		# Image variation can be performed with an image-to-text generation followed by a text-to-image generation:
		# 1. Image-to-text generation
		image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
		init_image = load_image(image_url).resize((512, 512))

		sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
		i2t_text = sample.text[0]
		print(i2t_text)

		# 2. Text-to-image generation
		sample = pipe(prompt=i2t_text, num_inference_steps=20, guidance_scale=8.0)
		final_image = sample.images[0]
		final_image.save("./unidiffuser_image_variation_sample.png")

	if True:
		# Text Variation

		model_id_or_path = "thu-ml/unidiffuser-v1"

		pipe = diffusers.UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
		pipe.to(device)

		# Text variation can be performed with a text-to-image generation followed by a image-to-text generation:
		# 1. Text-to-image generation
		prompt = "an elephant under the sea"

		sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
		t2i_image = sample.images[0]
		t2i_image.save("unidiffuser_text2img_sample_image.png")

		# 2. Image-to-text generation
		sample = pipe(image=t2i_image, num_inference_steps=20, guidance_scale=8.0)
		final_prompt = sample.text[0]
		print(final_prompt)

# REF [site] >> https://huggingface.co/Intel
def ldm3d_example():
	# Models:
	#	Intel/ldm3d
	#	Intel/ldm3d-4c
	#
	#	Intel/ldm3d-pano
	#
	#	Intel/ldm3d-sr

	import diffusers

	if True
		#model_id = "Intel/ldm3d"
		model_id = "Intel/ldm3d-4c"

		pipe = diffusers.StableDiffusionLDM3DPipeline.from_pretrained(model_id)
		#pipe.to("cpu")  # On CPU
		pipe.to("cuda")  # On GPU

		prompt = "A picture of some lemons on a table"
		name = "lemons"

		output = pipe(prompt)

		rgb_image, depth_image = output.rgb, output.depth
		#rgb_image[0].save(name + "_ldm3d_rgb.jpg")
		#depth_image[0].save(name + "_ldm3d_depth.png")
		rgb_image[0].save(name + "_ldm3d_4c_rgb.jpg")
		depth_image[0].save(name + "_ldm3d_4c_depth.png")

	if True:
		pipe = diffusers.StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-pano")
		#pipe.to("cpu")  # On CPU
		pipe.to("cuda")  # On GPU

		prompt = "360 view of a large bedroom"
		name = "bedroom_pano"

		output = pipe(
			prompt,
			width=1024,
			height=512,
			guidance_scale=7.0,
			num_inference_steps=50,
		) 

		rgb_image, depth_image = output.rgb, output.depth
		rgb_image[0].save(name+"_ldm3d_rgb.jpg")
		depth_image[0].save(name+"_ldm3d_depth.png")

	if True:
		# Generate a rgb/depth output from LDM3D
		pipe_ldm3d = diffusers.StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-4c")
		pipe_ldm3d.to("cuda")

		prompt = f"A picture of some lemons on a table"
		output = pipe_ldm3d(prompt)
		rgb_image, depth_image = output.rgb, output.depth
		rgb_image[0].save(f"lemons_ldm3d_rgb.jpg")
		depth_image[0].save(f"lemons_ldm3d_depth.png")

		# Upscale the previous output to a resolution of (1024, 1024)
		pipe_ldm3d_upscale = diffusers.DiffusionPipeline.from_pretrained("Intel/ldm3d-sr", custom_pipeline="pipeline_stable_diffusion_upscale_ldm3d")

		pipe_ldm3d_upscale.to("cuda")

		low_res_img = PIL.Image.open(f"lemons_ldm3d_rgb.jpg").convert("RGB")
		low_res_depth = PIL.Image.open(f"lemons_ldm3d_depth.png")
		outputs = pipe_ldm3d_upscale(prompt="high quality high resolution uhd 4k image", rgb=low_res_img, depth=low_res_depth, num_inference_steps=50, target_res=[1024, 1024])

		upscaled_rgb, upscaled_depth =outputs.rgb[0], outputs.depth[0]
		upscaled_rgb.save(f"upscaled_lemons_rgb.png")
		upscaled_depth.save(f"upscaled_lemons_depth.png")

def main():
	# Denoising diffusion.
	#	https://github.com/NVIDIA/modulus/tree/main/examples/generative/diffusion

	# Hugging Face Diffusers.
	#	https://github.com/huggingface/diffusers
	#	https://huggingface.co/docs/diffusers

	#diffusers_quickstart()
	#diffusers_intro()
	#diffusers_training_example()  # Not yet implemented
	#diffusers_dreambooth_training_example()  # DreamBooth. Not yet implemented

	#compvis_example()  # Stable diffusion
	stabilityai_example()  # Stable diffusion

	uni_diffuser_example()  # UniDiffuser

	# 3D
	#ldm3d_example()  # LDM3D

	#-----
	# Image synthesis
	#	Refer to ./image_synthesis_test.py

	# Video synthesis
	#	Refer to ./video_synthesis_test.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
