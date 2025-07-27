#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
import torch
import transformers

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv2
def layoutlmv2_example():
	import datasets

	if False:
		transformers.set_seed(88)

		processor = transformers.LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
		model = transformers.LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")

		dataset = datasets.load_dataset("hf-internal-testing/fixtures_docvqa")
		image_path = dataset["test"][0]["file"]
		image = Image.open(image_path).convert("RGB")
		encoding = processor(image, return_tensors="pt")

		outputs = model(**encoding)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Sequence classification.

		transformers.set_seed(88)

		dataset = datasets.load_dataset("rvl_cdip", split="train", streaming=True)
		data = next(iter(dataset))
		image = data["image"].convert("RGB")

		processor = transformers.LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
		model = transformers.LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes)

		encoding = processor(image, return_tensors="pt")
		sequence_label = torch.tensor([data["label"]])

		outputs = model(**encoding, labels=sequence_label)

		loss, logits = outputs.loss, outputs.logits
		predicted_idx = logits.argmax(dim=-1).item()
		predicted_answer = dataset.info.features["label"].names[predicted_idx]
		print(f"Predicted: index = {predicted_idx}, answer = {predicted_answer}.")

	if True:
		# Token classification.

		transformers.set_seed(88)

		datasets = datasets.load_dataset("nielsr/funsd", split="test")
		labels = datasets.features["ner_tags"].feature.names
		id2label = {v: k for v, k in enumerate(labels)}

		processor = transformers.LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
		model = transformers.LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=len(labels))

		data = datasets[0]
		image = Image.open(data["image_path"]).convert("RGB")
		words = data["words"]
		boxes = data["bboxes"]  # make sure to normalize your bounding boxes
		word_labels = data["ner_tags"]
		encoding = processor(
			image,
			words,
			boxes=boxes,
			word_labels=word_labels,
			padding="max_length",
			truncation=True,
			return_tensors="pt",
		)

		outputs = model(**encoding)

		logits, loss = outputs.logits, outputs.loss
		predicted_token_class_ids = logits.argmax(-1)
		predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
		print(f"Predicted token classes: {predicted_tokens_classes[:5]}.")

	if True:
		# Question answering.

		transformers.set_seed(88)

		processor = transformers.LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
		model = transformers.LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

		dataset = datasets.load_dataset("hf-internal-testing/fixtures_docvqa")
		image_path = dataset["test"][0]["file"]
		image = Image.open(image_path).convert("RGB")
		question = "When is coffee break?"
		encoding = processor(image, question, return_tensors="pt")

		outputs = model(**encoding)

		predicted_start_idx = outputs.start_logits.argmax(-1).item()
		predicted_end_idx = outputs.end_logits.argmax(-1).item()
		print(f"Predicted: start index = {predicted_start_idx}, end index = {predicted_end_idx}.")

		predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
		predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
		print(f"Predicted: answer = {predicted_answer}.")  # Results are not very good without further fine-tuning.

		#-----
		target_start_index = torch.tensor([7])
		target_end_index = torch.tensor([14])

		outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)

		predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
		predicted_answer_span_end = outputs.end_logits.argmax(-1).item()
		print(f"Predicted: answer span start = {predicted_answer_span_start}, answer span end = {predicted_answer_span_end}.")

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv3
def layoutlmv3_example():
	import datasets

	if False:
		processor = transformers.AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
		model = transformers.AutoModel.from_pretrained("microsoft/layoutlmv3-base")

		dataset = datasets.load_dataset("nielsr/funsd-layoutlmv3", split="train")
		example = dataset[0]
		image = example["image"]
		words = example["tokens"]
		boxes = example["bboxes"]
		encoding = processor(image, words, boxes=boxes, return_tensors="pt")

		outputs = model(**encoding)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Sequence classification.

		processor = transformers.AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
		model = transformers.AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

		dataset = datasets.load_dataset("nielsr/funsd-layoutlmv3", split="train")
		example = dataset[10]
		image = example["image"]
		words = example["tokens"]
		boxes = example["bboxes"]

		encoding = processor(image, words, boxes=boxes, return_tensors="pt")
		sequence_label = torch.tensor([1])

		outputs = model(**encoding, labels=sequence_label)

		loss = outputs.loss
		logits = outputs.logits
		#predicted_idx = logits.argmax(dim=-1).item()
		print(f"Loss = {loss}, logits = {logits}.")

	if True:
		# Token classification.

		processor = transformers.AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
		model = transformers.AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

		dataset = datasets.load_dataset("nielsr/funsd-layoutlmv3", split="train")
		example = dataset[0]
		image = example["image"]
		words = example["tokens"]
		boxes = example["bboxes"]
		word_labels = example["ner_tags"]

		encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

		outputs = model(**encoding)

		loss = outputs.loss
		logits = outputs.logits
		predicted_idx = logits.argmax(dim=-1)
		print(f"Predicted indices = {predicted_idx}.")

	if True:
		# Question answering.

		processor = transformers.AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
		model = transformers.AutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

		dataset = datasets.load_dataset("nielsr/funsd-layoutlmv3", split="train")
		example = dataset[0]
		image = example["image"]
		question = "what's his name?"
		words = example["tokens"]
		boxes = example["bboxes"]

		encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
		start_positions = torch.tensor([1])
		end_positions = torch.tensor([3])

		outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)

		loss = outputs.loss
		start_scores = outputs.start_logits
		end_scores = outputs.end_logits

		predicted_start_idx = start_scores.argmax(-1).item()
		predicted_end_idx = end_scores.argmax(-1).item()
		print(f"Predicted: start index = {predicted_start_idx}, end index = {predicted_end_idx}.")

		predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
		predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
		print(f"Predicted: answer = {predicted_answer}.")

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/donut
def donut_example():
	import re
	import datasets

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")
		model = transformers.DonutSwinModel.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Document image classification.

		processor = transformers.DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
		model.to(device)

		# Load document image.
		dataset = datasets.load_dataset("hf-internal-testing/example-documents", split="test")
		image = dataset[1]["image"]

		# Prepare decoder inputs.
		task_prompt = "<s_rvlcdip>"
		decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

		pixel_values = processor(image, return_tensors="pt").pixel_values

		outputs = model.generate(
			pixel_values.to(device),
			decoder_input_ids=decoder_input_ids.to(device),
			max_length=model.decoder.config.max_position_embeddings,
			early_stopping=True,
			pad_token_id=processor.tokenizer.pad_token_id,
			eos_token_id=processor.tokenizer.eos_token_id,
			use_cache=True,
			num_beams=1,
			bad_words_ids=[[processor.tokenizer.unk_token_id]],
			return_dict_in_generate=True,
		)

		sequence = processor.batch_decode(outputs.sequences)[0]
		sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
		sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token.
		print(processor.token2json(sequence))

	if True:
		# Document parsing.

		processor = transformers.DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
		model.to(device)

		# Load document image.
		dataset = datasets.load_dataset("hf-internal-testing/example-documents", split="test")
		image = dataset[2]["image"]

		# Prepare decoder inputs.
		task_prompt = "<s_cord-v2>"
		decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

		pixel_values = processor(image, return_tensors="pt").pixel_values

		outputs = model.generate(
			pixel_values.to(device),
			decoder_input_ids=decoder_input_ids.to(device),
			max_length=model.decoder.config.max_position_embeddings,
			early_stopping=True,
			pad_token_id=processor.tokenizer.pad_token_id,
			eos_token_id=processor.tokenizer.eos_token_id,
			use_cache=True,
			num_beams=1,
			bad_words_ids=[[processor.tokenizer.unk_token_id]],
			return_dict_in_generate=True,
		)

		sequence = processor.batch_decode(outputs.sequences)[0]
		sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
		sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token.
		print(processor.token2json(sequence))

	if True:
		# Document visual question answering.

		processor = transformers.DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
		model = transformers.VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
		model.to(device)

		# Load document image from the DocVQA dataset.
		dataset = datasets.load_dataset("hf-internal-testing/example-documents", split="test")
		image = dataset[0]["image"]

		# Prepare decoder inputs.
		task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
		question = "When is the coffee break?"
		prompt = task_prompt.replace("{user_input}", question)
		decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

		pixel_values = processor(image, return_tensors="pt").pixel_values

		outputs = model.generate(
			pixel_values.to(device),
			decoder_input_ids=decoder_input_ids.to(device),
			max_length=model.decoder.config.max_position_embeddings,
			early_stopping=True,
			pad_token_id=processor.tokenizer.pad_token_id,
			eos_token_id=processor.tokenizer.eos_token_id,
			use_cache=True,
			num_beams=1,
			bad_words_ids=[[processor.tokenizer.unk_token_id]],
			return_dict_in_generate=True,
		)

		sequence = processor.batch_decode(outputs.sequences)[0]
		sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
		sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token.
		print(processor.token2json(sequence))

# REF [site] >> https://medium.com/@unstructured-io/an-introduction-to-vision-transformers-for-document-understanding-e8aea045dd84
def donut_invoice_test():
	from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

	config_encoder = ViTConfig()
	config_decoder = BertConfig()
	config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
	model = VisionEncoderDecoderModel(config=config)

	#-----
	from donut.model import DonutModel

	model = DonutModel.from_pretrained("./custom-fine-tuned-model")

	prediction = model.inference(
		image=Image.open("./example-invoice.jpeg"), prompt="<s_dataset-donut-generated>"
	)["predictions"][0]

	print(prediction)

# REF [site] >>
#	https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer
#	https://huggingface.co/microsoft
def table_transformer_example():
	# Models:
	#	microsoft/table-transformer-detection: ~115MB.
	#	microsoft/table-transformer-structure-recognition: ~116MB.
	#	microsoft/table-transformer-structure-recognition-v1.1-all.
	#	microsoft/table-transformer-structure-recognition-v1.1-fin.
	#	microsoft/table-transformer-structure-recognition-v1.1-pub.

	import huggingface_hub
	from PIL import ImageDraw

	if False:
		# Initializing a Table Transformer microsoft/table-transformer-detection style configuration.
		configuration = transformers.TableTransformerConfig()

		# Initializing a model from the microsoft/table-transformer-detection style configuration.
		model = transformers.TableTransformerModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if False:
		file_path = huggingface_hub.hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
		image = Image.open(file_path).convert("RGB")

		image_processor = transformers.AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
		model = transformers.TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

		# Prepare image for the model.
		inputs = image_processor(images=image, return_tensors="pt")

		# Forward pass.
		outputs = model(**inputs)

		# The last hidden states are the final query embeddings of the Transformer decoder.
		last_hidden_states = outputs.last_hidden_state
		list(last_hidden_states.shape)  # (batch_size, num_queries, hidden_size).

	if True:
		file_path = huggingface_hub.hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
		image = Image.open(file_path).convert("RGB")

		# Table detection.
		#	Labels: {'table', 'table rotated'}.

		image_processor = transformers.AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
		model = transformers.TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

		inputs = image_processor(images=image, return_tensors="pt")
		outputs = model(**inputs)  # ['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'].

		# Convert outputs (bounding boxes and class logits) to COCO API.
		target_sizes = torch.tensor([image.size[::-1]])
		results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)  # ['scores', 'labels', 'boxes']*.

		image_draw = image.copy()
		draw = ImageDraw.Draw(image_draw)
		for result in results:
			for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
				box = [round(i, 2) for i in box.tolist()]
				print(f"Detected '{model.config.id2label[label.item()]}' with confidence {round(score.item(), 3)} at location {box}")
				draw.rectangle(box, outline=(255, 0, 0, 128), width=2)
		image_draw.show()
		#image_draw.save("./table_detection.png")

		#-----
		# Table structure recognition.
		#	Labels: {'table', 'table column', 'table row', 'table column header', 'table projected row header', 'table spanning cell'}.

		image = image.crop(results[0]["boxes"].detach().numpy().squeeze())  # One of detected tables.

		image_processor = transformers.AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
		model = transformers.TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

		inputs = image_processor(images=image, return_tensors="pt")
		outputs = model(**inputs)  # ['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'].

		# Convert outputs (bounding boxes and class logits) to COCO API.
		target_sizes = torch.tensor([image.size[::-1]])
		results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)  # ['scores', 'labels', 'boxes']*.

		image_draw = image.copy()
		draw = ImageDraw.Draw(image_draw)
		for result in results:
			for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
				box = [round(i, 2) for i in box.tolist()]
				print(f"Detected '{model.config.id2label[label.item()]}' with confidence {round(score.item(), 3)} at location {box}")
				#if model.config.id2label[label.item()] == "table":
				if model.config.id2label[label.item()] == "table column":
				#if model.config.id2label[label.item()] == "table row":
					draw.rectangle(box, outline=(255, 0, 0, 128), width=2)
		image_draw.show()
		#image_draw.save("./table_structure_recognition.png")

# REF [site] >> https://huggingface.co/microsoft
def tapex_example():
	# Models:
	#	microsoft/tapex-base: ~558MB.
	#	microsoft/tapex-large.
	#	microsoft/tapex-large-sql-execution.
	#
	#	microsoft/tapex-base-finetuned-wikisql.
	#	microsoft/tapex-base-finetuned-wtq: ~558MB.
	#	microsoft/tapex-large-finetuned-wikisql.
	#	microsoft/tapex-large-finetuned-wtq.
	#
	#	microsoft/tapex-base-finetuned-tabfact: ~560MB.
	#	microsoft/tapex-large-finetuned-tabfact.

	import pandas as pd

	if True:
		tokenizer = transformers.TapexTokenizer.from_pretrained("microsoft/tapex-base")
		model = transformers.BartForConditionalGeneration.from_pretrained("microsoft/tapex-base")

		data = {
			"year": [1896, 1900, 1904, 2004, 2008, 2012],
			"city": ["athens", "paris", "st. louis", "athens", "beijing", "london"],
		}
		table = pd.DataFrame.from_dict(data)

		# TaPEx accepts uncased input since it is pre-trained on the uncased corpus.
		query = "select year where city = beijing"
		encoding = tokenizer(table=table, query=query, return_tensors="pt")

		outputs = model.generate(**encoding)

		print(tokenizer.batch_decode(outputs, skip_special_tokens=True))  # Output: "['2008']".

	if True:
		tokenizer = transformers.TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
		model = transformers.BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")

		data = {
			"year": [1896, 1900, 1904, 2004, 2008, 2012],
			"city": ["athens", "paris", "st. louis", "athens", "beijing", "london"],
		}
		table = pd.DataFrame.from_dict(data)

		# TaPEx accepts uncased input since it is pre-trained on the uncased corpus.
		query = "In which year did beijing host the Olympic Games?"
		encoding = tokenizer(table=table, query=query, return_tensors="pt")

		outputs = model.generate(**encoding)

		print(tokenizer.batch_decode(outputs, skip_special_tokens=True))  # Output: "[' 2008.0']".

	if True:
		tokenizer = transformers.TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-tabfact")
		model = transformers.BartForSequenceClassification.from_pretrained("microsoft/tapex-base-finetuned-tabfact")

		data = {
			"year": [1896, 1900, 1904, 2004, 2008, 2012],
			"city": ["athens", "paris", "st. louis", "athens", "beijing", "london"],
		}
		table = pd.DataFrame.from_dict(data)

		# TaPEx accepts uncased input since it is pre-trained on the uncased corpus.
		query = "beijing hosts the olympic games in 2012"
		encoding = tokenizer(table=table, query=query, return_tensors="pt")

		outputs = model(**encoding)

		output_id = int(outputs.logits[0].argmax(dim=0))
		print(model.config.id2label[output_id])  # Outut: "Refused".

def korean_table_question_answering_example():
	from transformers import pipeline
	from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer
	import pandas as pd
	# REF [site] >> https://github.com/monologg/KoBERT-Transformers
	from tokenization_kobert import KoBertTokenizer

	data_dict = {
		"배우": ["송광호", "최민식", "설경구"],
		"나이": ["54", "58", "53"],
		"출연작품수": ["38", "32", "42"],
		"생년월일": ["1967/02/25", "1962/05/30", "1967/05/14"],
	}
	data_df = pd.DataFrame.from_dict(data_dict)

	if False:
		# Show the data frame.
		from IPython.display import display, HTML
		display(data_df)
		#print(HTML(data_df.to_html()).data)

	query = "최민식씨의 나이는?"

	# REF [site] >> https://huggingface.co/monologg
	pretrained_model_name = "monologg/kobert"
	#pretrained_model_name = "monologg/distilkobert"

	if False:
		# Not working.

		table_pipeline = pipeline(
			"table-question-answering",
			model=pretrained_model_name,
			tokenizer=KoBertTokenizer.from_pretrained(pretrained_model_name)
		)
	elif False:
		# Not working.

		#config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True, select_one_column=False)
		#model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name, config=config)
		model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name)

		table_pipeline = pipeline(
			"table-question-answering",
			model=model,
			tokenizer=KoBertTokenizer.from_pretrained(pretrained_model_name)
		)
	else:
		# Not correctly working.

		model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name)

		table_pipeline = pipeline(
			"table-question-answering",
			model=model,
			tokenizer=TapasTokenizer.from_pretrained(pretrained_model_name)
		)

	answer = table_pipeline(data_dict, query)
	#answer = table_pipeline(data_df, query)
	print("Answer: {}.".format(answer))

def main():
	# Document structure analysis

	# References:
	#	./paddle_ocr_test.py  # PaddleOCR
	#	./aru_net_test.py  # ARU-Net
	#	./dh_segment_test.py  # dhSegment
	
	# LayoutLM.
	#	Required libraries: tesseract, detectron2
	#layoutlmv2_example()  # LayoutLMv2
	#layoutlmv3_example()  # LayoutLMv3

	#-----
	# Document understanding

	# References:
	#	https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval
	#	https://developer.nvidia.com/blog/nvidia-nemo-retriever-delivers-accurate-multimodal-pdf-data-extraction-15x-faster/

	# Donut
	#donut_example()
	#donut_invoice_test()  # Not yet completed

	#-----
	# Text summarization

	# Refer to ./summa_test.py

	#-----
	# Table

	# References:
	#	./camelot_test.py
	#	./paddle_ocr_test.py  # PaddleOCR

	table_transformer_example()  # Table Transformer (TATR)
	#tapex_example()  # TaPEx

	#korean_table_question_answering_example()  # Not correctly working

	#-----
	# Augmentation

	# Refer to ./augraphy_test.py

	# Document image degradation
	#	./ocrodeg_test.py

#--------------------------------------------------------------------

if "_main__" == __name__:
	main()
