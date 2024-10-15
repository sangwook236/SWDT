#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video
def text_to_video_example() -> None:
	import torch
	import diffusers
	from diffusers.utils import export_to_video

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		if True:
			pipe = diffusers.DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
			pipe = pipe.to(device)
		elif False:
			pipe = diffusers.DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
			pipe.enable_model_cpu_offload()
			pipe.enable_vae_slicing()  # Memory optimization
		else:
			pipe = diffusers.DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
			pipe.scheduler = diffusers .DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
			pipe.enable_model_cpu_offload()

		prompt = "Spiderman is surfing"
		video_frames = pipe(prompt).frames[0]
		#video_frames = pipe(prompt, num_frames=64).frames[0]
		#video_frames = pipe(prompt, num_inference_steps=25).frames[0]
		video_path = export_to_video(video_frames)
		print(video_path)

	if True:
		from PIL import Image

		pipe = diffusers.DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
		pipe.enable_model_cpu_offload()

		# Memory optimization
		pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
		pipe.enable_vae_slicing()

		prompt = "Darth Vader surfing a wave"
		video_frames = pipe(prompt, num_frames=24).frames[0]
		video_path = export_to_video(video_frames)
		print(video_path)

		# Now the video can be upscaled
		pipe = diffusers.DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
		pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.enable_model_cpu_offload()

		# memory optimization
		pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
		pipe.enable_vae_slicing()

		video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

		video_frames = pipe(prompt, video=video, strength=0.6).frames[0]
		video_path = export_to_video(video_frames)
		print(video_path)

	if True:
		pipe = diffusers.TextToVideoSDPipeline.from_pretrained(
			"damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
		)
		pipe.enable_model_cpu_offload()

		prompt = "Spiderman is surfing"
		video_frames = pipe(prompt).frames[0]
		video_path = export_to_video(video_frames)
		print(video_path)

	if True:
		pipe = diffusers.DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
		pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.to("cuda")

		prompt = "spiderman running in the desert"
		video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]
		# Safe low-res video
		video_path = export_to_video(video_frames, output_video_path="./video_576_spiderman.mp4")

		# Let's offload the text-to-image model
		pipe.to("cpu")

		# and load the image-to-image model
		pipe = diffusers.DiffusionPipeline.from_pretrained(
			"cerspense/zeroscope_v2_XL", torch_dtype=torch.float16, revision="refs/pr/15"
		)
		pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.enable_model_cpu_offload()

		# The VAE consumes A LOT of memory, let's make sure we run it in sliced mode
		pipe.vae.enable_slicing()

		# Now let's upscale it
		video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

		# and denoise it
		video_frames = pipe(prompt, video=video, strength=0.6).frames[0]
		video_path = export_to_video(video_frames, output_video_path="./video_1024_spiderman.mp4")
		print(video_path)

# REF [site] >> https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video_zero
def text_to_video_zero_example() -> None:
	import numpy as np
	import torch
	import diffusers
	from PIL import Image
	import imageio

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		# Text-To-Video

		if True:
			model_id = "runwayml/stable-diffusion-v1-5"
			pipe = diffusers.TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
		else:
			# SDXL Support
			model_id = "stabilityai/stable-diffusion-xl-base-1.0"
			pipe = diffusers.TextToVideoZeroSDXLPipeline.from_pretrained(
				model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
			).to(device)

		prompt = "A panda is playing guitar on times square"
		result = pipe(prompt=prompt).images
		result = [(r * 255).astype("uint8") for r in result]
		imageio.mimsave("./video.mp4", result, fps=4)

	if True:
		# We can also generate longer videos by doing the processing in a chunk-by-chunk manner

		if True:
			model_id = "runwayml/stable-diffusion-v1-5"
			pipe = diffusers.TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
		else:
			# SDXL Support
			model_id = "stabilityai/stable-diffusion-xl-base-1.0"
			pipe = diffusers.TextToVideoZeroSDXLPipeline.from_pretrained(
				model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
			).to(device)
		seed = 0
		video_length = 24  # 24 ÷ 4fps = 6 seconds
		chunk_size = 8
		prompt = "A panda is playing guitar on times square"

		# Generate the video chunk-by-chunk
		result = []
		chunk_ids = np.arange(0, video_length, chunk_size - 1)
		generator = torch.Generator(device=device)
		for i in range(len(chunk_ids)):
			print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
			ch_start = chunk_ids[i]
			ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
			# Attach the first frame for Cross Frame Attention
			frame_ids = [0] + list(range(ch_start, ch_end))
			# Fix the seed for the temporal consistency
			generator.manual_seed(seed)
			output = pipe(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
			result.append(output.images[1:])

		# Concatenate chunks and save
		result = np.concatenate(result)
		result = [(r * 255).astype("uint8") for r in result]
		imageio.mimsave("./video.mp4", result, fps=4)

	if True:
		# Text-To-Video with Pose Control

		# 1. Download a demo video
		from huggingface_hub import hf_hub_download

		filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
		repo_id = "PAIR/Text2Video-Zero"
		video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

		# 2. Read video containing extracted pose images
		reader = imageio.get_reader(video_path, "ffmpeg")
		frame_count = 8
		pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

		# 3. Run StableDiffusionControlNetPipeline with our custom attention processor
		from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

		if True:
			model_id = "runwayml/stable-diffusion-v1-5"
			controlnet = diffusers.ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
			pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
				model_id, controlnet=controlnet, torch_dtype=torch.float16
			).to(device)

			# Set the attention processor
			pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
			pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

			# Fix latents for all frames
			latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)
		else:
			# SDXL Support
			controlnet_model_id = "thibaud/controlnet-openpose-sdxl-1.0"
			model_id = "stabilityai/stable-diffusion-xl-base-1.0"

			controlnet = diffusers.ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
			pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
				model_id, controlnet=controlnet, torch_dtype=torch.float16
			).to(device)

			# Set the attention processor
			pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
			pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

			# Fix latents for all frames
			latents = torch.randn((1, 4, 128, 128), device=device, dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

		prompt = "Darth Vader dancing in a desert"
		result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
		imageio.mimsave("./video.mp4", result, fps=4)

	if True:
		# Text-To-Video with Edge Control
		#	To perform text-guided video editing (with InstructPix2Pix)

		# 1. Download a demo video
		from huggingface_hub import hf_hub_download

		filename = "__assets__/pix2pix video/camel.mp4"
		repo_id = "PAIR/Text2Video-Zero"
		video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

		# 2. Read video from path
		reader = imageio.get_reader(video_path, "ffmpeg")
		frame_count = 8
		video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

		# 3. Run StableDiffusionInstructPix2PixPipeline with our custom attention processor
		from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

		model_id = "timbrooks/instruct-pix2pix"
		pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
		pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

		prompt = "make it Van Gogh Starry Night style"
		result = pipe(prompt=[prompt] * len(video), image=video).images
		imageio.mimsave("./edited_video.mp4", result, fps=4)

	if True:
		# DreamBooth specialization
		#	Methods Text-To-Video, Text-To-Video with Pose Control and Text-To-Video with Edge Control can run with custom DreamBooth models, as shown below for Canny edge ControlNet model and Avatar style DreamBooth model

		# 1. Download a demo video
		from huggingface_hub import hf_hub_download

		filename = "__assets__/canny_videos_mp4/girl_turning.mp4"
		repo_id = "PAIR/Text2Video-Zero"
		video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

		# 2. Read video from path
		reader = imageio.get_reader(video_path, "ffmpeg")
		frame_count = 8
		canny_edges = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

		# 3. Run StableDiffusionControlNetPipeline with custom trained DreamBooth model
		from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

		# Set model id to custom model
		model_id = "PAIR/text2video-zero-controlnet-canny-avatar"
		controlnet = diffusers.ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
		pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
			model_id, controlnet=controlnet, torch_dtype=torch.float16
		).to(device)

		# Set the attention processor
		pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
		pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

		# Fix latents for all frames
		latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)

		prompt = "oil painting of a beautiful girl avatar style"
		result = pipe(prompt=[prompt] * len(canny_edges), image=canny_edges, latents=latents).images
		imageio.mimsave("./video.mp4", result, fps=4)

# REF [site] >> https://huggingface.co/THUDM
def cog_video_test() -> None:
	# Models:
	#	THUDM/CogVideo
	#	THUDM/CogVideoX-2b
	#	THUDM/CogVideoX-5b
	#	THUDM/CogVideoX-5b-I2V  # Image-to-video

	# Install:
	#	pip install --upgrade transformers accelerate diffusers imageio-ffmpeg

	import torch

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if True:
		from diffusers import CogVideoXPipeline
		from diffusers.utils import export_to_video

		prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

		pipe = CogVideoXPipeline.from_pretrained(
			"THUDM/CogVideoX-2b",
			torch_dtype=torch.float16,
			#torch_dtype=torch.bfloat16,
		)

		pipe.enable_model_cpu_offload()
		pipe.enable_sequential_cpu_offload()
		pipe.vae.enable_slicing()
		pipe.vae.enable_tiling()

		video = pipe(
			prompt=prompt,
			num_videos_per_prompt=1,
			num_inference_steps=50,
			num_frames=49,
			guidance_scale=6,
			generator=torch.Generator(device=device).manual_seed(42),
		).frames[0]

		export_to_video(video, "./output.mp4", fps=8)

	if True:
		# Quantized inference

		from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline
		from diffusers.utils import export_to_video
		from transformers import T5EncoderModel
		from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

		quantization = int8_weight_only

		text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="text_encoder", torch_dtype=torch.bfloat16)
		quantize_(text_encoder, quantization())

		transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16)
		quantize_(transformer, quantization())

		vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.bfloat16)
		quantize_(vae, quantization())

		# Create pipeline and run inference
		pipe = CogVideoXPipeline.from_pretrained(
			"THUDM/CogVideoX-2b",
			text_encoder=text_encoder,
			transformer=transformer,
			vae=vae,
			torch_dtype=torch.bfloat16,
		)
		pipe.enable_model_cpu_offload()
		pipe.vae.enable_tiling()

		prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

		video = pipe(
			prompt=prompt,
			num_videos_per_prompt=1,
			num_inference_steps=50,
			num_frames=49,
			guidance_scale=6,
			generator=torch.Generator(device=device).manual_seed(42),
		).frames[0]

		export_to_video(video, "./output.mp4", fps=8)

	if True:
		from diffusers import CogVideoXImageToVideoPipeline
		from diffusers.utils import export_to_video, load_image

		prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
		image = load_image(image="./input.jpg")
		pipe = CogVideoXImageToVideoPipeline.from_pretrained(
			"THUDM/CogVideoX-5b-I2V",
			torch_dtype=torch.bfloat16,
		)

		pipe.enable_sequential_cpu_offload()
		pipe.vae.enable_tiling()
		pipe.vae.enable_slicing()

		video = pipe(
			prompt=prompt,
			image=image,
			num_videos_per_prompt=1,
			num_inference_steps=50,
			num_frames=49,
			guidance_scale=6,
			generator=torch.Generator(device=device).manual_seed(42),
		).frames[0]

		export_to_video(video, "./output.mp4", fps=8)

	if True:
		# Quantized inference

		from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
		from diffusers.utils import export_to_video, load_image
		from transformers import T5EncoderModel
		from torchao.quantization import quantize_, int8_weight_only

		quantization = int8_weight_only

		text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=torch.bfloat16)
		quantize_(text_encoder, quantization())

		transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V",subfolder="transformer", torch_dtype=torch.bfloat16)
		quantize_(transformer, quantization())

		vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
		quantize_(vae, quantization())

		# Create pipeline and run inference
		pipe = CogVideoXImageToVideoPipeline.from_pretrained(
			"THUDM/CogVideoX-5b-I2V",
			text_encoder=text_encoder,
			transformer=transformer,
			vae=vae,
			torch_dtype=torch.bfloat16,
		)

		pipe.enable_model_cpu_offload()
		pipe.vae.enable_tiling()
		pipe.vae.enable_slicing()

		prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
		image = load_image(image="./input.jpg")
		video = pipe(
			prompt=prompt,
			image=image,
			num_videos_per_prompt=1,
			num_inference_steps=50,
			num_frames=49,
			guidance_scale=6,
			generator=torch.Generator(device=device).manual_seed(42),
		).frames[0]

		export_to_video(video, "./output.mp4", fps=8)

def main():
	# Diffusion models
	#	Refer to ./diffusion_model_test.py

	text_to_video_example()
	text_to_video_zero_example()

	#cog_video_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
