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

# REF [function] >> donut_example() in hugging_face_transformers_test.py
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

	from PIL import Image
	import torch
	import transformers

	if True:
		# Install:
		#	pip install olmocr

		import base64
		import urllib.request
		from io import BytesIO

		from olmocr.data.renderpdf import render_pdf_to_base64png
		from olmocr.prompts import build_finetuning_prompt
		from olmocr.prompts.anchor import get_anchor_text

		# Initialize the model
		model = transformers.Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
		processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)

		# Grab a sample PDF
		urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", "./paper.pdf")

		# Render page 1 to an image
		image_base64 = render_pdf_to_base64png("./paper.pdf", 1, target_longest_image_dim=1024)

		# Build the prompt, using document metadata
		anchor_text = get_anchor_text("./paper.pdf", 1, pdf_engine="pdfreport", target_length=4000)
		prompt = build_finetuning_prompt(anchor_text)

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
		output = model.generate(
			**inputs,
			temperature=0.8,
			max_new_tokens=50,
			num_return_sequences=1,
			do_sample=True,
		)

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

	from openai import OpenAI
	import base64

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
	print(ocr_page_with_rolm(img_base64))

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
	#model_id = "google/paligemma-28b-mix-224"

	# Load the model, automatically moving it to the available device
	model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,  # Use bfloat16 for faster GPU inference if available
		device_map=device,
		revision="bfloat16" if device.type == "cuda" else "main"  # Use bfloat16 revision for GPU
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
	from PIL import Image
	from vllm import LLM

	import huggingface_hub
	huggingface_hub.login(token="<huggingface_token>")

	model_id = "google/paligemma-3b-mix-224"
	#model_id = "google/paligemma2-28b-mix-224"

	img_path = "path/to/image.png"

	llm = LLM(model=model_id)

	prompt = "ocr"
	image = Image.open(img_path)

	outputs = llm.generate({
		"prompt": prompt,
		"multi_modal_data": {
			"image": image
		},
	})

	for o in outputs:
		generated_text = o.outputs[0].text
		print(generated_text)

# REF [site] >> https://apidog.com/kr/blog/qwen-2-5-72b-open-source-ocr-kr/
def qwen_ocr_web_api_test():
	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Pull a model:
	#	ollama pull qwen2.5:72b

	import requests
	import base64

	# Encode an image to base64
	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")

	# Image
	image_path = "path/to/document.jpg"
	base64_image = encode_image(image_path)

	# OCR prompts
	#	일반 문서의 경우: "이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요."
	#	청구서의 경우: "청구서 번호, 날짜, 공급업체, 품목 및 총액을 포함하여 모든 청구서 세부 정보를 구조화된 JSON으로 추출하십시오."
	#	양식의 경우: "이 양식에서 모든 필드와 해당 값을 추출하고 JSON 형식으로 포맷하십시오."
	#	표의 경우: "이 표 데이터를 추출하고 JSON 배열 구조로 변환하십시오."

	# API request
	api_url = "<http://localhost:11434/api/generate>"
	payload = {
		"model": "qwen2.5:72b",
		"prompt": "이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요.",
		"images": [base64_image],
		"stream": False
	}

	# Send the request
	response = requests.post(api_url, json=payload)
	result = response.json()

	print(result["response"])
def qwen_ocr_python_test():
	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Pull a model:
	#	ollama pull qwen2.5:72b

	# Install:
	#	pip install ollama

	import base64, json
	import ollama

	# Encode an image to base64
	def encode_image(image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")

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
	response = client.generate(
		model="qwen2.5:72b",
		prompt="이 문서에서 텍스트를 추출하고 JSON 형식으로 포맷하세요",
		images=[base64_image],
		stream=False,
	)

	#result = response.json()  # Deprecated
	result = json.loads(response.model_dump_json())
	#print("Keys:", result.keys())

	#print(result)
	print(result["response"])

def main():
	#trocr_example()  # TrOCR

	#donut_ocr_test()  # Donut

	#-----
	# VLM OCR
	#	Mistral OCR (commercial)

	#olmocr_example()  # olmOCR (transformers). Use Qwen2-VL
	#rolmocr_example()  # RolmOCR (vLLM). Use Qwen2.5-VL

	#pali_gemma_ocr_transformers_test()  # PaliGemma (transformers)
	#pali_gemma_ocr_vllm_test()  # PaliGemma (vLLM)

	#qwen_ocr_web_api_test()  # Qwen2.5 (Ollama)
	qwen_ocr_python_test()  # Qwen2.5 (Ollama)

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
