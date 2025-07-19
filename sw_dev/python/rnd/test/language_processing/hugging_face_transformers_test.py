#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/huggingface/transformers
#	https://huggingface.co/transformers/
#	https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d

import requests, time
from PIL import Image
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertPreTrainedModel
from transformers import BertConfig
from transformers import *

# REF [site] >> https://github.com/huggingface/transformers
def quick_tour():
	# Transformers has a unified API for 10 transformer architectures and 30 pretrained weights.
	#          Model          | Tokenizer          | Pretrained weights shortcut
	MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
			  (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
			  (GPT2Model,       GPT2Tokenizer,       'gpt2'),
			  (CTRLModel,       CTRLTokenizer,       'ctrl'),
			  (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
			  (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
			  (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
			  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
			  (RobertaModel,    RobertaTokenizer,    'roberta-base'),
			  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
			 ]

	# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. 'TFRobertaModel' is the TF 2.0 counterpart of the PyTorch model 'RobertaModel'.

	# Let's encode some text in a sequence of hidden-states using each model.
	for model_class, tokenizer_class, pretrained_weights in MODELS:
		# Load pretrained model/tokenizer.
		tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
		model = model_class.from_pretrained(pretrained_weights)

		# Encode text.
		input_ids = torch.tensor([tokenizer.encode('Here is some text to encode', add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
		model.eval()
		with torch.no_grad():
			last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples.

	#--------------------
	# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
	BERT_MODEL_CLASSES = [
		BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
		BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering,
	]

	output_dir_path = './directory/to/save'
	import os
	os.makedirs(output_dir_path, exist_ok=True)

	# All the classes for an architecture can be initiated from pretrained weights for this architecture.
	# Note that additional weights added for fine-tuning are only initialized and need to be trained on the down-stream task.
	pretrained_weights = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
	for model_class in BERT_MODEL_CLASSES:
		# Load pretrained model/tokenizer.
		model = model_class.from_pretrained(pretrained_weights)

		# Models can return full list of hidden-states & attentions weights at each layer.
		model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
		input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
		all_hidden_states, all_attentions = model(input_ids)[-2:]

		# Models are compatible with Torchscript.
		model = model_class.from_pretrained(pretrained_weights, torchscript=True)
		traced_model = torch.jit.trace(model, (input_ids,))

		# Simple serialization for models and tokenizers.
		model.save_pretrained(output_dir_path)  # Save.
		model = model_class.from_pretrained(output_dir_path)  # Re-load.
		tokenizer.save_pretrained(output_dir_path)  # Save.
		tokenizer = BertTokenizer.from_pretrained(output_dir_path)  # Re-load.

		# SOTA examples for GLUE, SQUAD, text generation...

		print('{} processed.'.format(model_class.__name__))

def transformers_test():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	model = transformers.BertModel.from_pretrained("bert-base-uncased")

	max_length = 100
	#max_length = 1000
	gradient_accumulation_steps = 4
	#gradient_accumulation_steps = 40
	model_flops = (
		model.floating_point_ops({
			"input_ids": torch.zeros((1, max_length))
		})
		* gradient_accumulation_steps
	)

	#print("Model:")
	#print(model)
	print(f"Memory footprint = {model.get_memory_footprint() / 1e9} GB.")
	print(f"Flops = {model_flops / 1e9} GFLOPs.")

def tokenizer_test():
	pretrained_model_name = 'bert-base-uncased'
	#pretrained_model_name = 'openai-gpt'
	#pretrained_model_name = 'gpt2'

	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

	print(f'Type: {type(tokenizer)}.')
	print(f'Dir: {dir(tokenizer)}.')
	print(f'Vocab size = {tokenizer.vocab_size}.')
	print(f'Special tokens map: {tokenizer.special_tokens_map}.')
	print(f'SEP ID: {tokenizer.sep_token_id}, CLS ID: {tokenizer.cls_token_id}, MASK ID: {tokenizer.mask_token_id}.')
	print(f'BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}, UNK ID: {tokenizer.unk_token_id}.')
	print(f'BOS: {tokenizer.bos_token}, EOS: {tokenizer.eos_token}, PAD: {tokenizer.pad_token}, UNK: {tokenizer.unk_token}.')

	text = "Let's go to the movies tonight"
	encoding = tokenizer(text)
	#encoding = tokenizer(text, return_tensors='pt')  # {'tf', 'pt', 'np'}.
	print(f'Encoding keys = {encoding.keys()}.')  # {'input_ids', 'token_type_ids', 'attention_mask'}.
	print(f'Encoding: {encoding}.')

	print(f"Decoding: {tokenizer.decode(encoding['input_ids'])}.")
	print(f"Decoding: {tokenizer.decode(encoding['input_ids'], skip_special_tokens=True)}.")
	print(f"Decoding (each token): {[tokenizer.decode([token]) for token in encoding['input_ids']]}.")

	# Batch mode.
	texts = ["Let's go to the movies tonight", "Let's go to the movies today"]
	encoding = tokenizer(texts)
	print(f"Encoding #0: {encoding['input_ids'][0]}.")
	print(f"Encoding #1: {encoding['input_ids'][1]}.")
	print(f"Decoding (batch): {tokenizer.batch_decode(encoding['input_ids'])}.")
	print(f"Decoding (batch): {tokenizer.batch_decode(encoding['input_ids'], skip_special_tokens=True)}.")

	# Padding.
	#tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name, padding_side='left')
	tokenizer.padding_side = 'left'  # {'right', 'left'}.

	encoding = tokenizer(texts, padding='longest', return_tensors='pt')
	print(f"Input IDs: {encoding['input_ids']}.")
	print(f"Decoding (batch): {tokenizer.batch_decode(encoding['input_ids'])}.")

# REF [site] >> https://huggingface.co/docs/transformers/main_classes/pipelines
def pipeline_example():
	from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

	# Tasks:
	#	audio-classification: AudioClassificationPipeline.
	#	automatic-speech-recognition: AutomaticSpeechRecognitionPipeline.
	#	conversational: ConversationalPipeline.
	#	depth-estimation: DepthEstimationPipeline.
	#	document-question-answering: DocumentQuestionAnsweringPipeline.
	#	feature-extraction: FeatureExtractionPipeline.
	#	fill-mask: FillMaskPipeline.
	#	image-classification: ImageClassificationPipeline.
	#	image-segmentation: ImageSegmentationPipeline.
	#	image-to-text: ImageToTextPipeline.
	#	object-detection: ObjectDetectionPipeline.
	#	question-answering: QuestionAnsweringPipeline.
	#	summarization: SummarizationPipeline.
	#	table-question-answering: TableQuestionAnsweringPipeline.
	#	text2text-generation: Text2TextGenerationPipeline.
	#	text-classification (alias "sentiment-analysis" available): TextClassificationPipeline.
	#	text-generation: TextGenerationPipeline.
	#	token-classification (alias "ner" available): TokenClassificationPipeline.
	#	translation: TranslationPipeline.
	#	translation_xx_to_yy: TranslationPipeline.
	#	video-classification: VideoClassificationPipeline.
	#	visual-question-answering: VisualQuestionAnsweringPipeline.
	#	zero-shot-classification: ZeroShotClassificationPipeline.
	#	zero-shot-image-classification: ZeroShotImageClassificationPipeline.
	#	zero-shot-object-detection: ZeroShotObjectDetectionPipeline.

	# Sentiment analysis pipeline.
	sa_pipeline = pipeline('sentiment-analysis')

	# Question answering pipeline, specifying the checkpoint identifier.
	qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

	# Named entity recognition pipeline, passing in a specific model and tokenizer.
	# REF [site] >> https://huggingface.co/dbmdz
	model = AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
	tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
	ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

	#--------------------
	if False:
		"""
		conversation = Conversation('Going to the movies tonight - any suggestions?')

		# Steps usually performed by the model when generating a response:
		# 1. Mark the user input as processed (moved to the history)
		conversation.mark_processed()
		# 2. Append a mode response
		conversation.append_response('The Big lebowski.')

		conversation.add_user_input('Is it good?')
		"""

		conversational_pipeline = pipeline('conversational')

		conversation_1 = Conversation('Going to the movies tonight - any suggestions?')
		conversation_2 = Conversation("What's the last book you have read?")

		responses = conversational_pipeline([conversation_1, conversation_2])
		print('Responses:\n{}.'.format(responses))

		conversation_1.add_user_input('Is it an action movie?')
		conversation_2.add_user_input('What is the genre of this book?')

		responses = conversational_pipeline([conversation_1, conversation_2])
		print('Responses:\n{}.'.format(responses))

	#--------------------
	if False:
		if True:
			# Use BART in PyTorch.
			summarizer = pipeline('summarization')
		else:
			# Use T5 in TensorFlow.
			summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base', framework='tf')

		if False:
			#summary = summarizer('An apple a day, keeps the doctor away', min_length=5, max_length=20)
			summary = summarizer('An apple a day, keeps the doctor away', min_length=2, max_length=5)
		else:
			prompt = """
In physics, relativistic mechanics refers to mechanics compatible with special relativity (SR) and general relativity (GR).
It provides a non-quantum mechanical description of a system of particles, or of a fluid, in cases where the velocities of moving objects are comparable to the speed of light c.
As a result, classical mechanics is extended correctly to particles traveling at high velocities and energies, and provides a consistent inclusion of electromagnetism with the mechanics of particles.
This was not possible in Galilean relativity, where it would be permitted for particles and light to travel at any speed, including faster than light.
The foundations of relativistic mechanics are the postulates of special relativity and general relativity.
The unification of SR with quantum mechanics is relativistic quantum mechanics, while attempts for that of GR is quantum gravity, an unsolved problem in physics.
"""
			summary = summarizer(prompt, min_length=5, max_length=20)
		print('Summary: {}.'.format(summary))

	#--------------------
	# REF [site] >> https://huggingface.co/transformers/model_doc/tapas.html
	if False:
		import pandas as pd

		data_dict = {
			'actors': ['brad pitt', 'leonardo di caprio', 'george clooney'],
			'age': ['56', '45', '59'],
			'number of movies': ['87', '53', '69'],
			'date of birth': ['7 february 1967', '10 june 1996', '28 november 1967'],
		}
		data_df = pd.DataFrame.from_dict(data_dict)

		if False:
			# Show the data frame.
			from IPython.display import display, HTML
			display(data_df)
			#print(HTML(data_df.to_html()).data)

		query = 'How old is Brad Pitt?'
		#query = 'What is the age of Brad Pitt?'
		#query = 'How much is Brad PItt?'  # Incorrect question.

		table_pipeline = pipeline('table-question-answering')
		answer = table_pipeline(data_dict, query)
		#answer = table_pipeline(data_df, query)
		print('Answer: {}.'.format(answer))

	#--------------------
	if False:
		text2text_generator = pipeline('text2text-generation')
		generated = text2text_generator('question: What is 42 ? context: 42 is the answer to life, the universe and everything')
		print('Generated text: {}.'.format(generated))

def question_answering_example():
	from transformers import pipeline

	# Open and read the article.
	question = 'What is the capital of the Netherlands?'
	context = r"The four largest cities in the Netherlands are Amsterdam, Rotterdam, The Hague and Utrecht.[17] Amsterdam is the country's most populous city and nominal capital,[18] while The Hague holds the seat of the States General, Cabinet and Supreme Court.[19] The Port of Rotterdam is the busiest seaport in Europe, and the busiest in any country outside East Asia and Southeast Asia, behind only China and Singapore."

	# Generating an answer to the question in context.
	qa = pipeline(task='question-answering')
	answer = qa(question=question, context=context)

	# Print the answer.
	print(f'Question: {question}.')
	print(f"Answer: '{answer['answer']}' with score {answer['score']}.")

# REF [site] >> https://huggingface.co/krevas/finance-koelectra-small-generator
def korean_fill_mask_example():
	from transformers import pipeline

	# REF [site] >> https://huggingface.co/krevas
	fill_mask = pipeline(
		'fill-mask',
		model='krevas/finance-koelectra-small-generator',
		tokenizer='krevas/finance-koelectra-small-generator'
	)

	filled = fill_mask(f'내일 해당 종목이 대폭 {fill_mask.tokenizer.mask_token}할 것이다.')
	print(f'Filled mask: {filled}.')

# REF [site] >> https://huggingface.co/transformers/model_doc/encoderdecoder.html
def encoder_decoder_example():
	from transformers import EncoderDecoderConfig, EncoderDecoderModel
	from transformers import BertConfig, GPT2Config

	pretrained_model_name = 'bert-base-uncased'
	#pretrained_model_name = 'gpt2'

	if 'bert' in pretrained_model_name:
		# Initialize a BERT bert-base-uncased style configuration.
		config_encoder, config_decoder = BertConfig(), BertConfig()
	elif 'gpt2' in pretrained_model_name:
		config_encoder, config_decoder = GPT2Config(), GPT2Config()
	else:
		print('Invalid model, {}.'.format(pretrained_model_name))
		return

	config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

	if 'bert' in pretrained_model_name:
		# Initialize a Bert2Bert model from the bert-base-uncased style configurations.
		model = EncoderDecoderModel(config=config)
		#model = EncoderDecoderModel.from_encoder_decoder_pretrained(pretrained_model_name, pretrained_model_name)  # Initialize Bert2Bert from pre-trained checkpoints.
		tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
	elif 'gpt2' in pretrained_model_name:
		model = EncoderDecoderModel(config=config)
		tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)

	#print('Configuration of the encoder & decoder:\n{}.\n{}.'.format(model.config.encoder, model.config.decoder))
	#print('Encoder type = {}, decoder type = {}.'.format(type(model.encoder), type(model.decoder)))

	if False:
		# Access the model configuration.
		config_encoder = model.config.encoder
		config_decoder  = model.config.decoder

		# Set decoder config to causal LM.
		config_decoder.is_decoder = True
		config_decoder.add_cross_attention = True

	#--------------------
	input_ids = torch.tensor(tokenizer.encode('Hello, my dog is cute', add_special_tokens=True)).unsqueeze(0)  # Batch size 1.

	if False:
		# Forward.
		outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

		# Train.
		outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
		loss, logits = outputs.loss, outputs.logits

		# Save the model, including its configuration.
		model.save_pretrained('my-model')

		#-----
		# Load model and config from pretrained folder.
		encoder_decoder_config = EncoderDecoderConfig.from_pretrained('my-model')
		model = EncoderDecoderModel.from_pretrained('my-model', config=encoder_decoder_config)

	#--------------------
	# Generate.
	#	REF [site] >>
	#		https://huggingface.co/transformers/internal/generation_utils.html
	#		https://huggingface.co/blog/how-to-generate
	generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
	#generated = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, num_return_sequences=5, do_sample=True, top_k=0, temperature=0.7, early_stopping=True, decoder_start_token_id=model.config.decoder.pad_token_id)
	print('Generated = {}.'.format(tokenizer.decode(generated[0], skip_special_tokens=True)))

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/perceiver
#	https://huggingface.co/deepmind
def perceiver_example():
	# Models:
	#	deepmind/language-perceiver: ~805MB.
	#	deepmind/vision-perceiver-learned: ~249MB.
	#	deepmind/vision-perceiver-fourier: ~194MB.
	#	deepmind/vision-perceiver-conv: ~195MB.
	#	deepmind/optical-flow-perceiver: ~164MB.
	#	deepmind/multimodal-perceiver: ~79.5MB.

	if True:
		# EXAMPLE 1: using the Perceiver to classify texts
		# - we define a TextPreprocessor, which can be used to embed tokens
		# - we define a ClassificationDecoder, which can be used to decode the final hidden states of the latents to classification logits using trainable position embeddings
		config = transformers.PerceiverConfig()
		preprocessor = transformers.models.perceiver.modeling_perceiver.PerceiverTextPreprocessor(config)
		decoder = transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder(
			config,
			num_channels=config.d_latents,
			trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
			use_query_residual=True,
		)
		model = transformers.PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

		# You can then do a forward pass as follows:
		tokenizer = transformers.PerceiverTokenizer()
		text = "hello world"
		inputs = tokenizer(text, return_tensors="pt").input_ids

		with torch.no_grad():
			outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		# To train, one can train the model using standard cross-entropy:
		criterion = torch.nn.CrossEntropyLoss()

		labels = torch.tensor([1])
		loss = criterion(logits, labels)
		print(f"{loss=}.")

		# EXAMPLE 2: using the Perceiver to classify images
		# - we define an ImagePreprocessor, which can be used to embed images
		config = transformers.PerceiverConfig(image_size=224)
		preprocessor = transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor(
			config,
			prep_type="conv1x1",
			spatial_downsample=1,
			out_channels=256,
			position_encoding_type="trainable",
			concat_or_add_pos="concat",
			project_pos_dim=256,
			trainable_position_encoding_kwargs=dict(
				num_channels=256,
				index_dims=config.image_size**2,
			),
		)

		model = transformers.PerceiverModel(
			config,
			input_preprocessor=preprocessor,
			decoder=transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder(
				config,
				num_channels=config.d_latents,
				trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
				use_query_residual=True,
			),
		)

		# You can then do a forward pass as follows:
		image_processor = transformers.PerceiverImageProcessor()
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = image_processor(image, return_tensors="pt").pixel_values

		with torch.no_grad():
			outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		# To train, one can train the model using standard cross-entropy:
		criterion = torch.nn.CrossEntropyLoss()

		labels = torch.tensor([1])
		loss = criterion(logits, labels)
		print(f"{loss=}.")

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("deepmind/language-perceiver")
		model = transformers.PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")

		# Training
		text = "This is an incomplete sentence where some words are missing."
		inputs = tokenizer(text, padding="max_length", return_tensors="pt")
		# Mask " missing."
		inputs["input_ids"][0, 52:61] = tokenizer.mask_token_id
		labels = tokenizer(text, padding="max_length", return_tensors="pt").input_ids

		outputs = model(**inputs, labels=labels)
		loss = outputs.loss
		round(loss.item(), 2)

		logits = outputs.logits
		print(f"{logits.shape=}.")

		# Inference
		text = "This is an incomplete sentence where some words are missing."
		encoding = tokenizer(text, padding="max_length", return_tensors="pt")

		# Mask bytes corresponding to " missing.". Note that the model performs much better if the masked span starts with a space.
		encoding["input_ids"][0, 52:61] = tokenizer.mask_token_id

		# Forward pass
		with torch.no_grad():
			outputs = model(**encoding)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		masked_tokens_predictions = logits[0, 52:61].argmax(dim=-1).tolist()
		predicted = tokenizer.decode(masked_tokens_predictions)
		print(f"{predicted=}.")

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("deepmind/language-perceiver")
		model = transformers.PerceiverForSequenceClassification.from_pretrained("deepmind/language-perceiver")

		text = "hello world"
		inputs = tokenizer(text, return_tensors="pt").input_ids
		outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

	if False:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
		model = transformers. PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")

		inputs = image_processor(images=image, return_tensors="pt").pixel_values
		outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		# Model predicts one of the 1000 ImageNet classes
		predicted_class_idx = logits.argmax(-1).item()
		print("Predicted class:", model.config.id2label[predicted_class_idx])

	if False:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-fourier")
		model = transformers.PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier")

		inputs = image_processor(images=image, return_tensors="pt").pixel_values
		outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		# Model predicts one of the 1000 ImageNet classes
		predicted_class_idx = logits.argmax(-1).item()
		print("Predicted class:", model.config.id2label[predicted_class_idx])

	if False:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-conv")
		model = transformers.PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")

		inputs = image_processor(images=image, return_tensors="pt").pixel_values
		outputs = model(inputs=inputs)
		logits = outputs.logits
		print(f"{logits.shape=}.")

		# Model predicts one of the 1000 ImageNet classes
		predicted_class_idx = logits.argmax(-1).item()
		print("Predicted class:", model.config.id2label[predicted_class_idx])

	if False:
		model = transformers.PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

		# In the Perceiver IO paper, the authors extract a 3 x 3 patch around each pixel,
		# leading to 3 x 3 x 3 = 27 values for each pixel (as each pixel also has 3 color channels)
		# patches have shape (batch_size, num_frames, num_channels, height, width)
		# the authors train on resolutions of 368 x 496
		patches = torch.randn(1, 2, 27, 368, 496)
		outputs = model(inputs=patches)
		logits = outputs.logits
		print(f"{logits.shape=}.")

	if False:
		import numpy as np

		# Create multimodal inputs
		images = torch.randn((1, 16, 3, 224, 224))
		audio = torch.randn((1, 30720, 1))
		inputs = dict(image=images, audio=audio, label=torch.zeros((images.shape[0], 700)))

		model = transformers.PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver")

		# In the Perceiver IO paper, videos are auto-encoded in chunks
		# each chunk subsamples different index dimensions of the image and audio modality decoder queries
		nchunks = 128
		image_chunk_size = np.prod((16, 224, 224)) // nchunks
		audio_chunk_size = audio.shape[1] // model.config.samples_per_patch // nchunks
		# Process the first chunk
		chunk_idx = 0
		subsampling = {
			"image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
			"audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
			"label": None,
		}

		outputs = model(inputs=inputs, subsampled_output_points=subsampling)
		logits = outputs.logits
		print(f'{logits["audio"].shape=}.')
		print(f'{logits["image"].shape=}.')
		print(f'{logits["label"].shape=}.')

def main():
	# REF [site] >> https://huggingface.co/docs/transformers/index

	#import huggingface_hub
	#huggingface_hub.login(token="<huggingface_token>")

	#quick_tour()

	#transformers_test()
	#tokenizer_test()

	#--------------------
	# Pipeline.
	#	https://huggingface.co/docs/transformers/main/en/pipeline_tutorial

	#pipeline_example()

	#question_answering_example()

	#korean_fill_mask_example()

	#--------------------
	# Preprocessing.
	#	https://huggingface.co/docs/transformers/main/en/preprocessing

	#--------------------
	# Training.
	
	# Fine-tune a pretrained model:
	#	https://huggingface.co/docs/transformers/main/en/training
	# Distributed training with Hugging Face Accelerate:
	#	https://huggingface.co/docs/transformers/main/en/accelerate
	# Share a model:
	#	https://huggingface.co/docs/transformers/main/en/model_sharing

	# Fine-tune LLMs:
	#	Refer to ${SWR_HOME}/test/language_processing/llm_fine_tuning_test.py.

	#--------------------
	# Data and model parallelism.

	# Efficient Training on Multiple GPUs:
	#	https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many

	# Model Parallelism:
	#	https://huggingface.co/docs/transformers/v4.19.4/en/parallelism

	# Refer to llama2_example() in language_model_test.py.

	#--------------------
	# Models.

	encoder_decoder_example()
	#perceiver_example()  # Perceiver & Perceiver IO.

	# Refer to language_model_test.py.

#--------------------------------------------------------------------

if "_main__" == __name__:
	main()
