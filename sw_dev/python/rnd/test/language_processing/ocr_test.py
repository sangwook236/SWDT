#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/microsoft
def trocr_example():
	# Models:
	#	microsoft/trocr-base-handwritten
	#	microsoft/trocr-base-printed
	#	microsoft/trocr-base-stage1: ~1.54GB
	#	microsoft/trocr-small-handwritten
	#	microsoft/trocr-small-printed
	#	microsoft/trocr-small-stage1
	#	microsoft/trocr-large-handwritten
	#	microsoft/trocr-large-printed
	#	microsoft/trocr-large-stage1
	#	microsoft/trocr-base-str: ~1.34GB
	#	microsoft/trocr-large-str

	import requests
	from PIL import Image
	import torch
	import transformers

	if True:
		# Load image from the IIIT-5k dataset.
		url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
		image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

		processor = transformers.TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str")
		pixel_values = processor(images=image, return_tensors="pt").pixel_values

		generated_ids = model.generate(pixel_values)

		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		print(f"Generated text: {generated_text}.")

	if True:
		# Load image from the IAM database.
		url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
		image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

		processor = transformers.TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
		pixel_values = processor(images=image, return_tensors="pt").pixel_values

		generated_ids = model.generate(pixel_values)

		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		print(f"Generated text: {generated_text}.")

	if True:
		# Load image from the IAM database (actually this model is meant to be used on printed text).
		url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
		image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

		processor = transformers.TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
		pixel_values = processor(images=image, return_tensors="pt").pixel_values

		generated_ids = model.generate(pixel_values)

		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		print(f"Generated text: {generated_text}.")

	if True:
		# Load image from the IAM database.
		url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
		image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

		processor = transformers.TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

		# Training.
		pixel_values = processor(images=image, return_tensors="pt").pixel_values  # Batch size 1.
		decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
		outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)  # ['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions'].

# REF [site] >> https://github.com/deepseek-ai/DeepSeek-OCR
def deepseek_ocr_example():
	import os
	import torch
	import transformers

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	model_name = "deepseek-ai/DeepSeek-OCR"

	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	model = transformers.AutoModel.from_pretrained(model_name, attn_implementation="flash_attention_2", trust_remote_code=True, use_safetensors=True)
	model = model.eval().cuda().to(torch.bfloat16)

	#prompt = "<image>\nFree OCR. "
	prompt = "<image>\n<|grounding|>Convert the document to markdown. "
	image_file = "your_image.jpg"
	output_path = "your/output/dir"

	res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=1024, image_size=640, crop_mode=True, save_results=True, test_compress=True)

# REF [site] >>
#	https://huggingface.co/datalab-to
#	https://github.com/datalab-to/chandra
def chandra_example():
	# Models:
	#	datalab-to/chandra

	import transformers
	from chandra.model.hf import generate_hf
	from chandra.model.schema import BatchInputItem
	from chandra.output import parse_markdown
	from PIL import Image

	image_path = "path/to/image.jpg"
	PIL_IMAGE = Image.open(image_path)

	model = transformers.AutoModel.from_pretrained("datalab-to/chandra").cuda()
	model.processor = transformers.AutoProcessor.from_pretrained("datalab-to/chandra")

	batch = [
		BatchInputItem(
			image=PIL_IMAGE,
			prompt_type="ocr_layout"
		)
	]

	result = generate_hf(batch, model)[0]
	markdown = parse_markdown(result.raw)

# REF [function] >> donut_example() in document_processing_test.py
def donut_ocr_test():
	from PIL import Image
	import transformers

	# Step 1: Load the Donut model and processor
	def load_model():
		"""
		Load the Donut model and processor for OCR tasks.
		"""
		model_name = "naver-clova-ix/donut-base"
		processor = transformers.DonutProcessor.from_pretrained(model_name)
		model = transformers.VisionEncoderDecoderModel.from_pretrained(model_name)
		return processor, model

	# Step 2: Perform OCR using the Donut model
	def perform_ocr_with_llm(image_path, processor, model):
		"""
		Perform OCR on an image using the Donut model.
		"""
		try:
			# Load the image
			image = Image.open(image_path).convert("RGB")

			# Preprocess the image
			pixel_values = processor(image, return_tensors="pt").pixel_values

			# Generate text from the image
			outputs = model.generate(pixel_values, max_length=512, num_beams=4)
			generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

			return generated_text
		except Exception as ex:
			print(f"Error during OCR: {ex}")
			return None

	# Path to the input image
	image_path = "./sample_image.png"  # Replace with your image file

	# Load the model and processor
	print("Loading model...")
	processor, model = load_model()

	# Perform OCR
	print("Performing OCR...")
	extracted_text = perform_ocr_with_llm(image_path, processor, model)

	# Print the extracted text
	if extracted_text:
		print("\nExtracted Text:")
		print(extracted_text)

# REF [site] >> https://huggingface.co/allenai
def olmocr_example():
	# Models:
	#	allenai/olmOCR-7B-0225-preview
	#	allenai/olmOCR-7B-0225-preview-GGUF

	# Install:
	#	pip install olmocr

	import base64, time
	from io import BytesIO
	from PIL import Image
	import torch
	import transformers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")
	print(f"Number of available devices: {torch.cuda.device_count()}")

	# Initialize the model
	if False:
		model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
			"allenai/olmOCR-7B-0225-preview",
			torch_dtype=torch.bfloat16
		).eval()
		model.to(device)
	else:
		model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
			"allenai/olmOCR-7B-0225-preview",
			torch_dtype=torch.bfloat16,
			#torch_dtype="auto",  # Automatically use all available GPUs
			device_map="auto",  # Use all available GPUs
			#offload_folder="./offload",  # Add this line to enable CPU offloading
			#offload_state_dict=True,  # Add this line for state dict offloading
		).eval()
	processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

	if True:
		import urllib.request
		from olmocr.data.renderpdf import render_pdf_to_base64png
		from olmocr.prompts import build_finetuning_prompt
		from olmocr.prompts.anchor import get_anchor_text

		pdf_filepath: str = "./paper.pdf"

		# Grab a sample PDF
		urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", pdf_filepath)

		# Render page 1 to an image
		image_base64 = render_pdf_to_base64png(pdf_filepath, 1, target_longest_image_dim=1024)

		# Build the prompt, using document metadata
		anchor_text = get_anchor_text(pdf_filepath, 1, pdf_engine="pdfreport", target_length=4000)
		prompt = build_finetuning_prompt(anchor_text)
		#print(f"Prompt:\n{prompt}")
	else:
		# FIXME [check] >>

		# Encode an image to base64
		def encode_image(image_path):
			with open(image_path, "rb") as image_file:
				return base64.b64encode(image_file.read()).decode("utf-8")

		# Image
		image_path = "./ocr_data_240708/ocr_data_01_bar.png"
		image_base64 = encode_image(image_path)

		prompt = "ocr"

	# Build the full prompt
	messages = [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": prompt},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
			],
		}
	]

	# Apply the chat template and processor
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

	inputs = processor(
		text=[text],
		images=[main_image],
		padding=True,
		return_tensors="pt",
	)
	inputs = {key: value.to(device) for (key, value) in inputs.items()}

	# Generate the output
	print("Performing OCR...")
	start_time = time.time()
	output = model.generate(
		**inputs,
		temperature=0.8,
		max_new_tokens=50,
		num_return_sequences=1,
		do_sample=True,
	)
	print(f"OCR performed: {time.time() - start_time} secs.")

	# Decode the output
	prompt_length = inputs["input_ids"].shape[1]
	new_tokens = output[:, prompt_length:]
	text_output = processor.tokenizer.batch_decode(
		new_tokens, skip_special_tokens=True
	)

	print(text_output)
	# ['{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"natural_text":"Molmo and PixMo:\\nOpen Weights and Open Data\\nfor State-of-the']

# REF [site] >> https://huggingface.co/reducto
def rolmocr_example():
	# Models:
	#	reducto/RolmOCR

	# 1. Host your model with vLLM (OpenAI-compatible API):
	#	export VLLM_USE_V1=1
	#	vllm serve reducto/RolmOCR
	#	vllm serve reducto/RolmOCR --tensor-parallel-size 4

	# 2. Call the model via OpenAI-compatible server

	import base64, time
	from openai import OpenAI

	client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

	# This model is a fine-tuned version of Qwen/Qwen2.5-VL-7B-Instruct on the full allenai/olmOCR-mix-0225 dataset
	model = "reducto/RolmOCR"

	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")

	def ocr_page_with_rolm(img_base64):
		response = client.chat.completions.create(
			model=model,
			messages=[{
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{img_base64}"},
					},
					{
						"type": "text",
						"text": "Return the plain text representation of this document as if you were reading it naturally.\n",
					},
				],
			}],
			temperature=0.2,
			max_tokens=4096
		)
		return response.choices[0].message.content

	test_img_path = "path/to/image.png"
	img_base64 = encode_image(test_img_path)

	print("Performing OCR...")
	start_time = time.time()
	response = ocr_page_with_rolm(img_base64)
	print(f"OCR performed: {time.time() - start_time} secs.")

	print(response)

# REF [function] >> pali_gemma_example() in hugging_face_transformers_test.py
def pali_gemma_ocr_transformers_test():
	import requests, time
	from PIL import Image
	import torch
	import transformers

	# Check if CUDA is available and set the device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# 1. Load Model and Processor
	print("Loading model and processor...")
	start_time = time.time()
	# Model ID for PaliGemma
	# Other options include: google/paligemma-3b-pt-224, google/paligemma-3b-mix-448 etc.
	model_id = "google/paligemma-3b-mix-224"
	#model_id = "google/paligemma2-28b-mix-224"

	# Load the model, automatically moving it to the available device
	#quantization_config = transformers.TorchAoConfig("int4_weight_only", group_size=128)
	model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,  # Use bfloat16 for faster GPU inference if available
		device_map=device,
		revision="bfloat16" if device.type == "cuda" else "main"  # Use bfloat16 revision for GPU
		#quantization_config=quantization_config
	).eval()  # Set model to evaluation mode

	# Load the processor
	processor = transformers.AutoProcessor.from_pretrained(model_id)
	print(f"Model and processor loaded in {time.time() - start_time:.2f} seconds.")

	# 2. Prepare Image and Prompt
	# Image URL with text
	# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" #  A car with a license plate
	image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"  # A stop sign
	print(f"Loading image from: {image_url}")
	raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

	# Prompt asking the model to perform OCR-like task
	# For PaliGemma, specific prompts guide the task. "ocr" is used for reading text.
	prompt = "ocr"
	# Other prompts like "caption en", "detect box", "segment object" are also possible

	# 3. Process Inputs
	print("Processing inputs...")
	# The processor prepares the image and text prompt for the model
	inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
	# Adjust dtype if using GPU with bfloat16
	if device.type == "cuda":
		inputs = inputs.to(torch.bfloat16)

	input_len = inputs["input_ids"].shape[-1]  # Get the length of the input prompt tokens

	# 4. Generate Text (Perform OCR-like task)
	print("Generating text from image...")
	start_time = time.time()
	with torch.inference_mode():  # Use inference mode for efficiency
		# Generate text based on the image and prompt
		generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
		# Extract only the newly generated tokens, excluding the prompt
		generation = generation[0][input_len:]
		# Decode the generated tokens back into text
		decoded = processor.decode(generation, skip_special_tokens=True)
	print(f"Text generation finished in {time.time() - start_time:.2f} seconds.")

	# 5. Print Result
	print("\nPrompt:", prompt)
	print("Extracted Text:", decoded)

	# Display the image (optional)
	#import matplotlib.pyplot as plt
	#plt.imshow(raw_image)
	#plt.title("Input Image")
	#plt.axis("off")
	#plt.show()

# REF [site] >> https://docs.vllm.ai/en/v0.5.2/getting_started/examples/paligemma_example.html
def pali_gemma_ocr_vllm_test():
	import time
	from PIL import Image
	from vllm import LLM

	model_id = "google/paligemma-3b-mix-224"
	#model_id = "google/paligemma2-28b-mix-224"

	img_path = "path/to/image.png"

	llm = LLM(model=model_id)

	prompt = "ocr"
	image = Image.open(img_path)

	print("Performing OCR...")
	start_time = time.time()
	outputs = llm.generate({
		"prompt": prompt,
		"multi_modal_data": {
			"image": image
		},
	})
	print(f"OCR performed: {time.time() - start_time} secs.")

	for o in outputs:
		generated_text = o.outputs[0].text
		print(generated_text)

# REF [site] >> https://apidog.com/kr/blog/qwen-2-5-72b-open-source-ocr-kr/
def qwen_vl_ocr_web_api_test():
	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Run a model:
	#	ollama pull qwen2.5:72b
	#	ollama list
	#	ollama run qwen2.5:72b
	#	/bye
	#	ollama ps
	#	ollama stop qwen2.5:72b
	#	ollama rm qwen2.5:72b

	import requests, base64, time

	# Encode an image to base64
	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")

	#model_id = "qwen2.5:72b"  # Not working
	model_id = "qwen2.5vl"
	#model_id = "qwen2.5vl:72b"

	# Image
	image_path = "path/to/document.jpg"
	base64_image = encode_image(image_path)

	# OCR prompts
	#	일반 문서의 경우: "이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요."
	#	청구서의 경우: "청구서 번호, 날짜, 공급업체, 품목 및 총액을 포함하여 모든 청구서 세부 정보를 구조화된 JSON으로 추출하십시오."
	#	양식의 경우: "이 양식에서 모든 필드와 해당 값을 추출하고 JSON 형식으로 포맷하십시오."
	#	표의 경우: "이 표 데이터를 추출하고 JSON 배열 구조로 변환하십시오."

	# API request
	api_url = "http://localhost:11434/api/generate"
	payload = {
		"model": model_id,
		"prompt": "이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요.",
		"images": [base64_image],
		#"images": [image_path],  # Error: illegal base64 data at input byte 0
		"stream": False
	}

	# Send the request
	print("Performing OCR...")
	start_time = time.time()
	response = requests.post(api_url, json=payload)
	print(f"OCR performed: {time.time() - start_time} secs.")

	result = response.json()
	if "response" in result:
		print(result["response"])
	else:
		print(result)

def qwen_vl_ocr_python_test():
	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Run a model:
	#	ollama pull qwen2.5:72b
	#	ollama list
	#	ollama run qwen2.5:72b
	#	/bye
	#	ollama ps
	#	ollama stop qwen2.5:72b
	#	ollama rm qwen2.5:72b

	# Install:
	#	pip install ollama

	import base64, json, time
	import ollama

	# Encode an image to base64
	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")

	#model_id = "qwen2.5:72b"  # Not working
	model_id = "qwen2.5vl"
	#model_id = "qwen2.5vl:72b"

	# Image
	image_path = "path/to/document.jpg"
	base64_image = encode_image(image_path)

	client = ollama.Client(
		host="http://localhost:11434",
		#headers={"x-some-header": "some-value"},
	)

	# OCR prompts
	#	일반 문서의 경우: "이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요."
	#	청구서의 경우: "청구서 번호, 날짜, 공급업체, 품목 및 총액을 포함하여 모든 청구서 세부 정보를 구조화된 JSON으로 추출하십시오."
	#	양식의 경우: "이 양식에서 모든 필드와 해당 값을 추출하고 JSON 형식으로 포맷하십시오."
	#	표의 경우: "이 표 데이터를 추출하고 JSON 배열 구조로 변환하십시오."

	# Generate
	print("Performing OCR...")
	start_time = time.time()
	response = client.generate(
		model=model_id,
		prompt="이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요",
		images=[base64_image],
		#images=[image_path],
		stream=False,
	)
	print(f"OCR performed: {time.time() - start_time} secs.")

	#result = response.json()  # Deprecated
	result = json.loads(response.model_dump_json())
	#print("Keys:", result.keys())

	#print(result)
	if "response" in result:
		print(result["response"])
	else:
		print(result)

# REF [site] >> https://github.com/TechExpertTutorials/Meta-LLama-OCR
def llama_ocr_example():
	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Run a model:
	#	ollama pull llama3.2-vision
	#	ollama list
	#	ollama run llama3.2-vision
	#	/bye
	#	ollama ps
	#	ollama stop llama3.2-vision
	#	ollama rm llama3.2-vision

	if True:
		import requests, base64, time

		SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""

		def encode_image_to_base64(image_path):
			"""Convert an image file to a base64 encoded string."""
			with open(image_path, "rb") as image_file:
				return base64.b64encode(image_file.read()).decode("utf-8")

		model_id = "llama3.2-vision"
		image_path = "./images/WalmartReceipt.png"
		#image_path = "./images/dl1.jpg"
		#image_path = "./images/dl2.png"

		base64_image = encode_image_to_base64(image_path)
		payload = {
			"model": model_id,
			#"prompt": "What is in this image?",
			"prompt": SYSTEM_PROMPT,
			"images": [base64_image],
			"stream": False
		}

		try:
			print("Performing OCR...")
			start_time = time.time()
			response = requests.post("http://localhost:11434/api/generate", json=payload)
			response.raise_for_status()
			print(f"OCR performed: {time.time() - start_time} secs.")

			#print(response.json()["response"])
			print(response.json())
		except requests.exceptions.RequestException as ex:
			print(f"Error querying Ollama: {ex}")
		except KeyError:
			print("Error: 'response' key not found in Ollama response.")

	if False:
		# Install:
		#	pip install ollama

		import time
		import ollama

		SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""

		model_id = "llama3.2-vision"
		image_path = "./images/WalmartReceipt.png"
		#image_path = "./images/dl1.jpg"
		#image_path = "./images/dl2.png"

		print("Performing OCR...")
		start_time = time.time()
		response = ollama.chat(
			model=model_id,
			messages=[{
				"role": "user",
				#"content": "What is in this image?",
				"content": SYSTEM_PROMPT,
				"images": [image_path]
			}]
		)
		print(f"OCR performed: {time.time() - start_time} secs.")

		print(response)

# REF [site] >> https://github.com/TechExpertTutorials/GemmaOCR
def gemma_ocr_example():
	import time
	from PIL import Image
	import torch
	import transformers

	model_id = "google/gemma-3-4b-it"

	image = "./images/WalmartReceipt.png"
	prompt = "What items are on this receipt, what is the cost of each and what is the total cost and what is the date on the receipt?"

	model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
		model_id, device_map="auto"
	).eval()

	processor = transformers.AutoProcessor.from_pretrained(model_id)

	messages = [
		{
			"role": "system",
			"content": [{"type": "text", "text": "You are a helpful assistant."}]
		},
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image},
				{"type": "text", "text": prompt}
			]
		}
	]

	inputs = processor.apply_chat_template(
		messages, add_generation_prompt=True, tokenize=True,
		return_dict=True, return_tensors="pt"
	).to(model.device, dtype=torch.bfloat16)

	input_len = inputs["input_ids"].shape[-1]

	with torch.inference_mode():
		print("Performing OCR...")
		start_time = time.time()
		generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
		print(f"OCR performed: {time.time() - start_time} secs.")

		generation = generation[0][input_len:]

	decoded = processor.decode(generation, skip_special_tokens=True)
	print(decoded)

# REF [site] >> https://github.com/TechExpertTutorials/Phi3-OCR
def phi_3_ocr_example():
	import time
	from PIL import Image
	import transformers

	model_id = "microsoft/Phi-3.5-vision-instruct"

	if True:
		images = [Image.open("./images/WalmartReceipt.png")]
		placeholder = "<|image_1|>\n"
		messages = [
			{"role": "user", "content": placeholder + "Extract all the text you can find on this image. Give me the address of the store, date of the receipt, the names of the items and the cost for each. also give me the total cost"},
		]
	elif False:
		images = [
			Image.open("./images/Declaration_P1.png"),
			Image.open("./images/Declaration_P2.png"),
		]
		placeholder = "<|image_1|>\n<|image_2|>\n"
		messages = [
			{"role": "user", "content": placeholder + "give me some quotes from both of these pages. Output should be formatted like this: Image 1: Quotes\nImage 2: Quotes"},
		]
	elif False:
		images = [
			Image.open("./images/Declaration_P1.png"),
			Image.open("./images/Declaration_P2.png"),
		]
		placeholder = "<|image_1|>\n<|image_2|>\n"
		messages = [
			{"role": "user", "content": placeholder +" give me 10 short quotes of 10 words or less from both pages"},
		]

	# Note: set _attn_implementation="eager" if you don't have flash_attn installed
	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype="auto",
		device_map="auto",
		#device_map="cuda",
		trust_remote_code=True,
		_attn_implementation="flash_attention_2"
	)

	# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
	processor = transformers.AutoProcessor.from_pretrained(
		model_id,
		trust_remote_code=True,
		num_crops=4
	) 

	prompt = processor.tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True
	)

	#inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
	inputs = processor(prompt, images, return_tensors="pt").to("cuda")

	generation_args = {
		#"max_new_tokens": 10000,
		"max_new_tokens": 100,
		#"temperature": 0.0,  # Choose temperature or do_sample
		"do_sample": False,
	} 

	print("Performing OCR...")
	start_time = time.time()
	generate_ids = model.generate(
		**inputs,
		eos_token_id=processor.tokenizer.eos_token_id,
		**generation_args
	)
	print(f"OCR performed: {time.time() - start_time} secs.")

	# Remove input tokens 
	generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
	response = processor.batch_decode(
		generate_ids,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False
	)[0]

	print(response)

# REF [site] >> https://github.com/TechExpertTutorials/Phi4OCR
def phi_4_ocr_example():
	# Install:
	#	pip install peft backoff flash_attn

	import time
	from PIL import Image
	import transformers

	message = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. Provide only the transcription without any additional comments."""

	image_input = "./images/WalmartReceipt.png"

	# Specify the model checkpoint
	model_name = "microsoft/Phi-4-multimodal-instruct"

	generation_config = transformers.GenerationConfig.from_pretrained(model_name)

	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype="auto",
		device_map="auto",
		#device_map="cuda",
		trust_remote_code=True,
		_attn_implementation="flash_attention_2",
	).eval().cuda()

	processor = transformers.AutoProcessor.from_pretrained(
		model_name,
		trust_remote_code=True,
	)

	# Example image URL containing text
	try:
		image = Image.open(image_input).convert("RGB")
	except Exception as ex:
		print(f"Error loading image: {ex}")
		exit()

	# Prepare the input
	prompt = f"<|user|><|image_1|>{message}<|end|><|assistant|>"
	inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

	# Generate the OCR output
	try:
		print("Performing OCR...")
		start_time = time.time()
		#outputs = model.generate(**inputs, max_new_tokens=500)  # Adjust max_new_tokens as needed
		outputs = model.generate(**inputs, max_new_tokens=1000, generation_config=generation_config)
		print(f"OCR performed: {time.time() - start_time} secs.")

		predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
		print("Extracted Text:")
		print(predicted_text)
	except Exception as ex:
		print(f"Error during generation: {ex}")

# REF [site] >> https://github.com/TechExpertTutorials/QwenOCR
def qwen_vl_ocr_example():
	# Install:
	#	pip install qwen-vl-utils

	import time
	import torch
	import transformers
	from qwen_vl_utils import process_vision_info

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")
	print(f"Number of available devices: {torch.cuda.device_count()}")

	#model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
	model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

	model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained( 
		model_id, torch_dtype="auto", device_map="auto"
		#model_id, torch_dtype="auto", device_map="cpu"
		#model_id, torch_dtype="auto", device_map="cuda"
	)
	processor = transformers.AutoProcessor.from_pretrained(model_id, use_fast=True)

	#image = r"./images/dl1.jpg"
	#prompt = "Extract all text found on the image, including handwritten signatures"
	image = r"./images/WalmartReceipt.png"
	prompt = "Extract all text found on the image, including handwritten signatures"
	#image = r"./images/WalmartReceipt.png"
	#prompt = "What is the account number shown on this image?"

	messages = [{
		"role": "user",
		"content": [
			{"type": "image", "image": image},
			{"type": "text", "text": prompt},
		],
	}]

	# Preparation for inference
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
	#inputs = inputs.to("cuda")

	#model = model.to("cuda")

	print("Performing OCR...")
	start_time = time.time()
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	#generated_ids = model.generate(**inputs, max_new_tokens=1024)
	print(f"OCR performed: {time.time() - start_time} secs.")

	generated_ids_trimmed = [
		out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]

	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)

	print(output_text)

# REF [site] >>
#	https://github.com/TechExpertTutorials/DeepSeek-VL2-OCR
#	https://github.com/TechExpertTutorials/DeepSeek-VL2-OCR-Linux
def deepseek_vl_ocr_example():
	# Install:
	#	pip install torch=2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	#	pip install deepseek_vl2
	#	pip install -e .

	import time
	from PIL import Image
	import torch
	import transformers
	from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
	from deepseek_vl2.utils.io import load_pil_images

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# Automatic mixed precision - lowers precision (FP16 or BF16) to use less memory
	import torch.cuda.amp as amp
	scaler = amp.GradScaler()  # Initialize a scaler

	# Specify the path to the model
	#model_path = "deepseek-ai/deepseek-vl2"
	#model_path = "deepseek-ai/deepseek-vl2-small"
	model_path = "deepseek-ai/deepseek-vl2-tiny"
	vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
	tokenizer = vl_chat_processor.tokenizer

	vl_gpt: DeepseekVLV2ForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
	#vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
	vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

	if True:
		# Single image conversation example

		images = [Image.open("./images/animals.jpg").convert("RGB")]

		# Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
		# If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
		conversation = [
			{
				"role": "<|User|>",
				"content": "<image>\n<|ref|>Describe the image.<|/ref|>.",
				"images": images,
			},
			{"role": "<|Assistant|>", "content": ""},
		]
	else:
		# Multiple images (or in-context learning) conversation example

		images = [
			Image.open("./images/dog_a.png").convert("RGB"),
			Image.open("./images/dog_b.png").convert("RGB"),
			Image.open("./images/dog_c.png").convert("RGB"),
			Image.open("./images/dog_d.png").convert("RGB"),
		]

		conversation = [
			{
				"role": "User",
				"content": "A dog wearing nothing in the foreground, "
					"a dog wearing a santa hat, "
					"a dog wearing a wizard outfit, and "
					"what's the dog wearing?",
				"images": images,
			},
			{"role": "Assistant", "content": ""}
		]

	# Load images and prepare for inputs
	model_inputs = vl_chat_processor(
		conversations=conversation,
		images=images,
		force_batchify=True,
		system_prompt=""
	).to(vl_gpt.device)

	# Run image encoder to get the image embeddings
	inputs_embeds = vl_gpt.prepare_inputs_embeds(**model_inputs)

	# Run the model to get the response
	with torch.no_grad():
		print("Performing OCR...")
		start_time = time.time()
		outputs = vl_gpt.language.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=model_inputs.attention_mask,
			pad_token_id=tokenizer.eos_token_id,
			bos_token_id=tokenizer.bos_token_id,
			eos_token_id=tokenizer.eos_token_id,
			max_new_tokens=2048,
			do_sample=False,
			use_cache=True
		)
		print(f"OCR performed: {time.time() - start_time} secs.")

	#answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
	#print(f'{model_inputs["sft_format"][0]}', answer)
	response = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
	print(response)

def phi_3_ocr_test():
	import time
	from PIL import Image
	import torch
	import transformers

	image_path = "path/to/document_image.png"

	#model_name = "microsoft/Phi-3.5-vision-128k-instruct"  # Error
	model_name = "microsoft/Phi-3.5-vision-instruct"

	processor = transformers.AutoProcessor.from_pretrained(
		model_name,
		trust_remote_code=True,
	)
	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch.float16,
		#torch_dtype="auto",
		device_map="auto",
		trust_remote_code=True,
	)

	def extract_text(image_path):
		"""Extract all text from a document image."""
		images = [Image.open(image_path)]

		prompt = "<|image_1|>\nExtract all text from this document image. Return only the extracted text."

		inputs = processor(
			text=prompt,
			images=images,
			return_tensors="pt"
		).to(model.device)

		with torch.no_grad():
			output = model.generate(
				**inputs,
				max_new_tokens=1024,
				do_sample=False,
			)

		extracted_text = processor.decode(output[0], skip_special_tokens=True)
		# Remove the prompt from the response
		extracted_text = extracted_text.replace(prompt, "").strip()

		return extracted_text

	def extract_key_value_pairs(image_path):
		"""Extract key-value pairs from tables or text in the document image."""
		images = [Image.open(image_path)]

		prompt = "<|image_1|>\nExtract all key-value pairs from this document image. Format the output as JSON with keys and values."

		inputs = processor(
			text=prompt,
			images=images,
			return_tensors="pt"
		).to(model.device)

		with torch.no_grad():
			output = model.generate(
				**inputs,
				max_new_tokens=1024,
				do_sample=False
			)

		extracted_pairs = processor.decode(output[0], skip_special_tokens=True)
		# Remove the prompt from the response
		extracted_pairs = extracted_pairs.replace(prompt, "").strip()

		return extracted_pairs

	# Extract text
	print("Extracting text...")
	start_time = time.time()
	text = extract_text(image_path)
	print(f"Text extracted: {time.time() - start_time} secs.")

	print(text)
	
	# Extract key-value pairs
	start_time = time.time()
	print("Extracting key-value pairs...")
	kv_pairs = extract_key_value_pairs(image_path)
	print(f"Key-value pairs extracted: {time.time() - start_time} secs.")

	print(kv_pairs)

def main():
	#trocr_example()  # TrOCR
	deepseek_ocr_example()  # DeepSeek-OCR
	#chandra_example()  # Chandra

	#-----
	# Document OCR

	# Refer to doctr_example() in ./document_processing_test.py
	# Refer to surya_example() in ./document_processing_test.py

	#donut_ocr_test()  # Donut

	#-----
	# VLM OCR
	#	Mistral OCR (commercial)

	# For Hugging Face models
	#import huggingface_hub
	#huggingface_hub.login(token="<huggingface_token>")

	#olmocr_example()  # olmOCR (transformers). Use Qwen2-VL
	#rolmocr_example()  # RolmOCR (vLLM). Use Qwen2.5-VL

	#pali_gemma_ocr_transformers_test()  # PaliGemma (transformers)
	#pali_gemma_ocr_vllm_test()  # PaliGemma (vLLM)

	#qwen_vl_ocr_web_api_test()  # Qwen2.5-VL (Ollama)
	#qwen_vl_ocr_python_test()  # Qwen2.5-VL (Ollama)

	#llama_ocr_example()  # Llama 3.2 Vision (Ollama)
	#gemma_ocr_example()  # Gemma 3 (transformers)
	#phi_3_ocr_example()  # Phi-3.5-vision (transformers)
	#phi_4_ocr_example()  # Phi-4-multimodal (transformers)
	#qwen_vl_ocr_example()  # Qwen2.5-VL (transformers)
	#deepseek_vl_ocr_example()  # DeepSeek-VL2 (transformers)

	#phi_3_ocr_test()  # Phi-3.5-vision (transformers)

	#-----
	# NeMo Retriever OCR
	#	https://build.nvidia.com/nvidia/nemoretriever-ocr-v1
	#	https://docs.nvidia.com/nim/ingestion/image-ocr/latest/api-reference.html

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
