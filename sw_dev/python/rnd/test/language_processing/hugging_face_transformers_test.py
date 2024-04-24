#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/huggingface/transformers
#	https://huggingface.co/transformers/
#	https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d

import requests, time
import torch
from PIL import Image
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

def korean_table_question_answering_example():
	from transformers import pipeline
	from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer
	import pandas as pd
	# REF [site] >> https://github.com/monologg/KoBERT-Transformers
	from tokenization_kobert import KoBertTokenizer

	data_dict = {
		'배우': ['송광호', '최민식', '설경구'],
		'나이': ['54', '58', '53'],
		'출연작품수': ['38', '32', '42'],
		'생년월일': ['1967/02/25', '1962/05/30', '1967/05/14'],
	}
	data_df = pd.DataFrame.from_dict(data_dict)

	if False:
		# Show the data frame.
		from IPython.display import display, HTML
		display(data_df)
		#print(HTML(data_df.to_html()).data)

	query = '최민식씨의 나이는?'

	# REF [site] >> https://huggingface.co/monologg
	pretrained_model_name = 'monologg/kobert'
	#pretrained_model_name = 'monologg/distilkobert'

	if False:
		# Not working.

		table_pipeline = pipeline(
			'table-question-answering',
			model=pretrained_model_name,
			tokenizer=KoBertTokenizer.from_pretrained(pretrained_model_name)
		)
	elif False:
		# Not working.

		#config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True, select_one_column=False)
		#model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name, config=config)
		model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name)

		table_pipeline = pipeline(
			'table-question-answering',
			model=model,
			tokenizer=KoBertTokenizer.from_pretrained(pretrained_model_name)
		)
	else:
		# Not correctly working.

		model = TapasForQuestionAnswering.from_pretrained(pretrained_model_name)

		table_pipeline = pipeline(
			'table-question-answering',
			model=model,
			tokenizer=TapasTokenizer.from_pretrained(pretrained_model_name)
		)

	answer = table_pipeline(data_dict, query)
	#answer = table_pipeline(data_df, query)
	print('Answer: {}.'.format(answer))

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

def gpt2_example():
	# NOTE [info] >> Refer to example codes in the comment of forward() of each BERT class in https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py

	pretrained_model_name = 'gpt2'
	tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)

	input_ids = torch.tensor(tokenizer.encode('Hello, my dog is cute', add_special_tokens=True)).unsqueeze(0)  # Batch size 1.

	if True:
		print('Loading a model...')
		start_time = time.time()
		# The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.
		model = GPT2Model.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple.
		print('{} processed.'.format(GPT2Model.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
		model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids, labels=input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		loss, logits = outputs[:2]
		print('{} processed.'.format(GPT2LMHeadModel.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for RocStories/SWAG tasks.
		model = GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		# Add a [CLS] to the vocabulary (we should train it also!).
		tokenizer.add_special_tokens({'cls_token': '[CLS]'})
		model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size.
		print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary.

		choices = ['Hello, my dog is cute [CLS]', 'Hello, my cat is cute [CLS]']
		encoded_choices = [tokenizer.encode(s) for s in choices]
		cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
		input_ids0 = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2.
		mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1.

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids0, mc_token_ids=mc_token_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		lm_prediction_scores, mc_prediction_scores = outputs[:2]
		print('{} processed.'.format(GPT2DoubleHeadsModel.__name__))

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def sentence_completion_model_using_gpt2_example():
	# Load pre-trained model tokenizer (vocabulary).
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

	# Encode a text inputs.
	text = 'What is the fastest car in the'
	indexed_tokens = tokenizer.encode(text)

	# Convert indexed tokens in a PyTorch tensor.
	tokens_tensor = torch.tensor([indexed_tokens])
	# If you have a GPU, put everything on cuda.
	tokens_tensor = tokens_tensor.to('cuda')

	# Load pre-trained model (weights).
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	model.to('cuda')

	# Predict all tokens.
	model.eval()  # Set the model in evaluation mode to deactivate the DropOut modules.
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]

	# Get the predicted next sub-word.
	predicted_index = torch.argmax(predictions[0, -1, :]).item()
	predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

	# Print the predicted word.
	print(f'Predicted text = {predicted_text}.')

# REF [site] >>
#	https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
#		python pytorch-transformers/examples/run_generation.py --model_type=gpt2 --length=100 --model_name_or_path=gpt2
#	https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def conditional_text_generation_using_gpt2_example():
	raise NotImplementedError

# REF [site] >> https://huggingface.co/EleutherAI
def eleuther_ai_gpt_test():
	# Models:
	#	EleutherAI/gpt-j-6B: ~24.2GB. Too big to load.
	#	EleutherAI/gpt-neo-125M: ~526MB.
	#	EleutherAI/gpt-neo-1.3B: ~5.31GB.
	#	EleutherAI/gpt-neo-2.7B: ~10.7GB.
	#	EleutherAI/gpt-neox-20B: ~42.6GB. Too big to load.

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}.')

	# Text generation.
	if False:
		from transformers import pipeline

		# REF [site] >> https://huggingface.co/docs/transformers/model_doc/gpt_neo
		#pretrained_model_name = 'EleutherAI/gpt-neo-125M'
		#pretrained_model_name = 'EleutherAI/gpt-neo-1.3B'
		pretrained_model_name = 'EleutherAI/gpt-neo-2.7B'

		generator = pipeline(task='text-generation', model=pretrained_model_name, device=device)

		prompt = 'EleutherAI has'
		min_new_tokens, max_new_tokens = 50, 500
		print('Generating text...')
		start_time = time.time()
		gen_texts = generator(prompt, do_sample=True, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
		print(f'Text generated: {time.time() - start_time} secs.')

		assert len(gen_texts) == 1
		print('Generated text:')
		print(gen_texts[0]['generated_text'])

	#-----
	# Text generation.
	if False:
		from transformers import AutoTokenizer, AutoModelForCausalLM

		# REF [site] >> https://huggingface.co/docs/transformers/model_doc/gptj
		#pretrained_model_name = 'EleutherAI/gpt-j-6B'
		# REF [site] >> https://huggingface.co/docs/transformers/model_doc/gpt_neox
		pretrained_model_name = 'EleutherAI/gpt-neox-20b'

		tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
		model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
		#model.to(device=device, non_blocking=True)

		prompt = 'EleutherAI has'
		tokens = tokenizer.encode(prompt, return_tensors='pt')
		#tokens = tokens.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			min_new_tokens, max_new_tokens = 50, 500

			print('Generating text...')
			start_time = time.time()
			# REF [site] >> https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation
			#gen_tokens = model.generate(tokens, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Greedy decoding.
			#gen_tokens = model.generate(tokens, penalty_alpha=0.5, top_k=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Contrastive searching.
			#gen_tokens = model.generate(tokens, num_beams=3, no_repeat_ngram_size=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search decoding.
			#gen_tokens = model.generate(tokens, num_beams=2, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search decoding.
			#gen_tokens = model.generate(tokens, num_beams=5, num_beam_groups=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Diverse beam search decoding.
			gen_tokens = model.generate(tokens, do_sample=True, temperature=0.9, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, num_beams=1, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Multinomial sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, num_beams=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search multinomial sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, top_k=10, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Top-k sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, top_p=0.8, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Top-p sampling.
			print(f'Text generated: {time.time() - start_time} secs.')

			gen_texts = tokenizer.batch_decode(gen_tokens)
			assert len(gen_texts) == 1
			print('Generated text:')
			print(gen_texts[0])

	#--------------------
	# Question answering.
	if True:
		from transformers import AutoTokenizer, AutoModelForCausalLM

		pretrained_model_name = 'EleutherAI/gpt-neo-2.7B'
		#pretrained_model_name = 'EleutherAI/gpt-j-6B'

		tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
		model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
		model.to(device)

		query = 'What is the capital of Tamilnadu?'
		inputs = tokenizer.encode(f'Q: {query}\nA:', return_tensors='pt').to(device)

		outputs = model.generate(inputs, max_length=1024, do_sample=True)

		answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
		#print(answer)
		answer = answer.split('A:')[1].strip()
		print(f'Answer: {answer}.')

# REF [site] >>
#	https://github.com/SKT-AI/KoGPT2
#	https://huggingface.co/skt/kogpt2-base-v2
def skt_gpt_test():
	# Models:
	#	skt/kogpt2-base-v2: ~490MB.

	from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}.')

	pretrained_model_name = 'skt/kogpt2-base-v2'

	tokenizer = PreTrainedTokenizerFast.from_pretrained(
		pretrained_model_name,
		bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
	)
	tokenized_text = tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
	print(f'Tokenized text: {tokenized_text}.')

	model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
	model.to(device=device, non_blocking=True)

	text = '근육이 커지기 위해서는'
	input_ids = tokenizer.encode(text, return_tensors='pt')
	input_ids = input_ids.to(device=device, non_blocking=True)

	model.eval()
	with torch.no_grad():
		print('Generating text...')
		start_time = time.time()
		gen_ids = model.generate(
			input_ids,
			max_length=128,
			repetition_penalty=2.0,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
			bos_token_id=tokenizer.bos_token_id,
			use_cache=True
		)
		print(f'Text generated: {time.time() - start_time} secs.')

		assert len(gen_ids) == 1
		generated = tokenizer.decode(gen_ids[0])
		print('Generated text:')
		print(generated)

# REF [site] >>
#	https://huggingface.co/kakaobrain/kogpt
#	https://github.com/kakaobrain/kogpt
def kakao_brain_gpt_test():
	# Models:
	#	kakaobrain/kogpt:
	#		KoGPT6B-ryan1.5b: ~24.7GB.
	#		KoGPT6B-ryan1.5b-float16: ~12.3GB.

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}.')

	pretrained_model_name = 'kakaobrain/kogpt'
	#revision = 'KoGPT6B-ryan1.5b'
	revision = 'KoGPT6B-ryan1.5b-float16'

	# Text generation.
	if False:
		from transformers import AutoTokenizer, AutoModelForCausalLM

		tokenizer = AutoTokenizer.from_pretrained(
			pretrained_model_name, revision=revision,
			bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
		)
		model = AutoModelForCausalLM.from_pretrained(
			pretrained_model_name, revision=revision,
			pad_token_id=tokenizer.eos_token_id, torch_dtype='auto', low_cpu_mem_usage=True
		)
		model.to(device=device, non_blocking=True)

		prompt = "인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던"
		tokens = tokenizer.encode(prompt, return_tensors='pt')
		tokens = tokens.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			min_new_tokens, max_new_tokens = 50, 500

			print('Generating text...')
			start_time = time.time()
			# REF [site] >> https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation
			#gen_tokens = model.generate(tokens, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Greedy decoding.
			#gen_tokens = model.generate(tokens, penalty_alpha=0.5, top_k=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Contrastive searching.
			#gen_tokens = model.generate(tokens, num_beams=3, no_repeat_ngram_size=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search decoding.
			#gen_tokens = model.generate(tokens, num_beams=2, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search decoding.
			#gen_tokens = model.generate(tokens, num_beams=5, num_beam_groups=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Diverse beam search decoding.
			gen_tokens = model.generate(tokens, do_sample=True, temperature=0.9, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, num_beams=1, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Multinomial sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, num_beams=3, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Beam search multinomial sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, top_k=10, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Top-k sampling.
			#gen_tokens = model.generate(tokens, do_sample=True, top_p=0.8, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Top-p sampling.
			print(f'Text generated: {time.time() - start_time} secs.')

			gen_texts = tokenizer.batch_decode(gen_tokens)
			assert len(gen_texts) == 1
			print('Generated text:')
			print(gen_texts[0])

	# Text generation.
	if False:
		from transformers import pipeline

		generator = pipeline(task='text-generation', model=pretrained_model_name, revision=revision)
		#generator = pipeline(task='text-generation', model=pretrained_model_name, revision=revision, device=device)  # torch.cuda.OutOfMemoryError: CUDA out of memory.

		prompt = "인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던"
		min_new_tokens, max_new_tokens = 50, 500

		print('Generating text...')
		start_time = time.time()
		gen_texts = generator(prompt, do_sample=True, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
		print(f'Text generated: {time.time() - start_time} secs.')

		assert len(gen_texts) == 1
		print('Generated text:')
		print(gen_texts[0]['generated_text'])

	#--------------------
	# Text summarization.
	if True:
		from transformers import AutoTokenizer, AutoModelForCausalLM

		tokenizer = AutoTokenizer.from_pretrained(
			pretrained_model_name, revision=revision,
			bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
		)
		model = AutoModelForCausalLM.from_pretrained(
			pretrained_model_name, revision=revision,
			#torch_dtype=torch.float16, low_cpu_mem_usage=True,
			pad_token_id=tokenizer.eos_token_id, torch_dtype='auto', low_cpu_mem_usage=True
		)
		model.to(device=device, non_blocking=True)

		prompt1 = """
상대성 이론(相對性理論 / Theory of Relativity)은 알베르트 아인슈타인이 주장한 인간, 생물, 행성, 항성, 은하 크기 이상의 거시 세계를 다루는 이론이다.
양자역학과 함께 우주에 기본적으로 작용하는 법칙을 설명하는 이론이자 현대 물리학에서 우주를 이해하는 데 사용하는 두 개의 가장 근본적인 이론이다.
시간과 공간을 시공간으로, 물질과 에너지를 통합하는 데에 성공해 빛과 어둠을 인류에게 가져다 주었다.
E=mc^2 이 바로 특수 상대성 이론에서 제시된 질량-에너지 등가 방정식이다.
특수상대성이론에 따르면 관성 좌표계(system)에서 물리 법칙은 동일하게 적용되며, 관찰자나 광원의 속도에 관계없이 진공 중에서 진행하는 빛의 속도는 일정하며 빛보다 빠른 건 없다.
이에 따라 시간과 공간은 속도에 따라 상대적이다.
또 일반상대성이론에 따르면 가속 좌표계에서 중력과 관성력은 본질적으로 같은 것이고 강한 중력은 시공간을 휘게 하며 정지한 쪽의 시간이 더 길게 간다.
한 줄 요약:
"""
		prompt2 = """
KoGPT는 방대한 한국어 데이터로 훈련된 인공지능(Artifical Intelligence, AI) 한국어 언어 모델입니다.
카카오브레인의 GPT-3 기반 언어 모델로써 한국어에 대해 다양한 과제를 수행할 수 있습니다.
KoGPT API는 KoGPT가 제공하는 기능을 REST API 방식으로 손쉽게 사용할 수 있는 서비스입니다.
KoGPT API는 사람처럼 맥락을 이해하고 문제를 해결합니다.
상품 소개글 작성, 감정 분석, 기계 독해, 기계 번역 등 높은 수준의 언어 과제를 해결할 수 있어 다양한 분야에서 활용 가능합니다.
한 줄 요약:
"""
		prompt3 = """
과학기술정보통신부와 한국항공우주연구원은 지난 27일 저녁 6시에 다누리의 달 궤도 진입 최종 성공을 확인했다고 28일 밝혔다.
다누리가 달 궤도 진입에 최종 성공함에 따라 우리나라는 달 궤도선을 개발해 달까지 도달할 수 있는 진정한 우주탐사 역량을 확보했다.
또 앞으로 달 착륙선 등 후속 우주탐사를 추진할 수 있는 기반을 마련한 것이다.
"""
		tokens = tokenizer.encode(prompt3, return_tensors='pt')
		tokens = tokens.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			min_length, max_length = 5, 20

			print('Summarizing text...')
			start_time = time.time()
			summary = model(tokens)  # Failed to summarize.
			print(f'Text summarized: {time.time() - start_time} secs.')

			#print(f'Loss: {summary.loss}.')  # None.
			print(f'Logits: shape = {summary.logits.shape}, dtype = {summary.logits.dtype}.')
			summary = summary.logits.cpu().argmax(axis=-1)
			summary = tokenizer.batch_decode(summary)  # FIXME [check] >>
			assert len(summary) == 1
			print('Summary:')
			print(summary[0])

	# Text summarization.
	if False:
		from transformers import pipeline

		if True:
			summarizer = pipeline(task='summarization', model=pretrained_model_name, revision=revision)
			#summarizer = pipeline(task='summarization', model=pretrained_model_name, revision=revision, device=device)  # torch.cuda.OutOfMemoryError: CUDA out of memory.
		else:
			from transformers import AutoTokenizer, AutoModelForCausalLM

			tokenizer = AutoTokenizer.from_pretrained(
				pretrained_model_name, revision=revision,
				bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
			)
			model = AutoModelForCausalLM.from_pretrained(
				pretrained_model_name, revision=revision,
				pad_token_id=tokenizer.eos_token_id, torch_dtype='auto', low_cpu_mem_usage=True
			)
			summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
			#summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, device=device)  # torch.cuda.OutOfMemoryError: CUDA out of memory.

		prompt1 = """
상대성 이론(相對性理論 / Theory of Relativity)은 알베르트 아인슈타인이 주장한 인간, 생물, 행성, 항성, 은하 크기 이상의 거시 세계를 다루는 이론이다.
양자역학과 함께 우주에 기본적으로 작용하는 법칙을 설명하는 이론이자 현대 물리학에서 우주를 이해하는 데 사용하는 두 개의 가장 근본적인 이론이다.
시간과 공간을 시공간으로, 물질과 에너지를 통합하는 데에 성공해 빛과 어둠을 인류에게 가져다 주었다.
E=mc^2 이 바로 특수 상대성 이론에서 제시된 질량-에너지 등가 방정식이다.
특수상대성이론에 따르면 관성 좌표계(system)에서 물리 법칙은 동일하게 적용되며, 관찰자나 광원의 속도에 관계없이 진공 중에서 진행하는 빛의 속도는 일정하며 빛보다 빠른 건 없다.
이에 따라 시간과 공간은 속도에 따라 상대적이다.
또 일반상대성이론에 따르면 가속 좌표계에서 중력과 관성력은 본질적으로 같은 것이고 강한 중력은 시공간을 휘게 하며 정지한 쪽의 시간이 더 길게 간다.
"""
		prompt2 = """
KoGPT는 방대한 한국어 데이터로 훈련된 인공지능(Artifical Intelligence, AI) 한국어 언어 모델입니다.
카카오브레인의 GPT-3 기반 언어 모델로써 한국어에 대해 다양한 과제를 수행할 수 있습니다.
KoGPT API는 KoGPT가 제공하는 기능을 REST API 방식으로 손쉽게 사용할 수 있는 서비스입니다.
KoGPT API는 사람처럼 맥락을 이해하고 문제를 해결합니다.
상품 소개글 작성, 감정 분석, 기계 독해, 기계 번역 등 높은 수준의 언어 과제를 해결할 수 있어 다양한 분야에서 활용 가능합니다.
"""
		prompt3 = """
과학기술정보통신부와 한국항공우주연구원은 지난 27일 저녁 6시에 다누리의 달 궤도 진입 최종 성공을 확인했다고 28일 밝혔다.
다누리가 달 궤도 진입에 최종 성공함에 따라 우리나라는 달 궤도선을 개발해 달까지 도달할 수 있는 진정한 우주탐사 역량을 확보했다.
또 앞으로 달 착륙선 등 후속 우주탐사를 추진할 수 있는 기반을 마련한 것이다.
"""
		min_length, max_length = 5, 20

		print('Summarizing text...')
		start_time = time.time()
		summary = summarizer(prompt3, min_length=min_length, max_length=max_length)  # Failed to summarize.
		print(f'Text summarized: {time.time() - start_time} secs.')

		assert len(summary) == 1
		print('Summary:')
		print(summary[0]['summary_text'])

# REF [site] >>
#	https://github.com/nomic-ai/gpt4all
#	https://github.com/nomic-ai/nomic
def gpt4all_example():
	# Models:
	#	nomic-ai/gpt4all-j.
	#	nomic-ai/gpt4all-lora.
	#	nomic-ai/gpt4all-j-lora.

	import datasets

	if True:
		model = transformers.AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
	elif False:
		dataset = datasets.load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision="v1.2-jazzy")
		model = transformers.AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j-prompt-generations", revision="v1.2-jazzy")

	#--------------------
	# Install:
	#	pip install nomic

	import nomic.gpt4all

	if False:
		# CPU.

		m = nomic.gpt4all.GPT4All()
		m.open()

		response = m.prompt("write me a story about a lonely computer")
		print(f"{response=}.")
	elif False:
		# GPU.

		LLAMA_PATH = ...

		m = nomic.gpt4all.GPT4AllGPU(LLAMA_PATH)
		config = {
			"num_beams": 2,
			"min_new_tokens": 10,
			"max_length": 100,
			"repetition_penalty": 2.0,
		}

		response = m.generate("write me a story about a lonely computer", config)
		print(f"{response=}.")
	elif False:
		# REF [site] >> https://github.com/nomic-ai/nomic/blob/main/examples/GPT4All.ipynb

		response = nomic.gpt4all.gpt4all.prompt("Tell me a story about a bear who becomes friends with a bunny.")
		print(f"{response=}.")
	elif False:
		# REF [site] >> https://github.com/nomic-ai/nomic/blob/main/examples/GPT4All.ipynb

		with nomic.gpt4all.GPT4All() as session:
			prompts = ["Hello, there. I have a couple requests.", "First: tell me a joke.", "What's the largest city in the United States"]
			for prompt in prompts:
				print("------PROMPT-------\n" + prompt)
				response = session.prompt(prompt)
				print("-----RESPONSE------\n" + response)
	elif True:
		# REF [site] >> https://github.com/nomic-ai/nomic/blob/main/examples/GPT4All.ipynb

		session = nomic.gpt4all.GPT4All()

		session.open()
		response = session.prompt("How are you doing today?")
		print(response)
		response = session.prompt("Oh really? Why is that?")
		print(response)
		session.close()

def bert_example():
	# NOTE [info] >> Refer to example codes in the comment of forward() of each BERT class in https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py

	pretrained_model_name = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

	input_ids = torch.tensor(tokenizer.encode('Hello, my dog is cute', add_special_tokens=True)).unsqueeze(0)  # Batch size 1.

	if True:
		print('Loading a model...')
		start_time = time.time()
		# The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
		model = BertModel.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple.
		print('{} processed.'.format(BertModel.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with two heads on top as done during the pre-training: a 'masked language modeling' head and a 'next sentence prediction (classification)' head.
		model = BertForPreTraining.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		prediction_scores, seq_relationship_scores = outputs[:2]
		print('{} processed.'.format(BertForPreTraining.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with a 'language modeling' head on top.
		model = BertForMaskedLM.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids, masked_lm_labels=input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		loss, prediction_scores = outputs[:2]
		print('{} processed.'.format(BertForMaskedLM.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with a 'next sentence prediction (classification)' head on top.
		model = BertForNextSentencePrediction.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		seq_relationship_scores = outputs[0]
		print('{} processed.'.format(BertForNextSentencePrediction.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.
		model = BertForSequenceClassification.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1.

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids, labels=labels)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		loss, logits = outputs[:2]
		print('{} processed.'.format(BertForSequenceClassification.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.
		model = BertForMultipleChoice.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		choices = ['Hello, my dog is cute', 'Hello, my cat is amazing']
		input_ids0 = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices.
		labels = torch.tensor(1).unsqueeze(0)  # Batch size 1.

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids0, labels=labels)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		loss, classification_scores = outputs[:2]
		print('{} processed.'.format(BertForMultipleChoice.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.
		model = BertForTokenClassification.from_pretrained(pretrained_model_name)
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1.

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			outputs = model(input_ids, labels=labels)
		print('Inferred: {} secs.'.format(time.time() - start_time))

		loss, scores = outputs[:2]
		print('{} processed.'.format(BertForTokenClassification.__name__))

	if True:
		print('Loading a model...')
		start_time = time.time()
		# Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute 'span start logits' and 'span end logits').
		model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		print('A model loaded: {} secs.'.format(time.time() - start_time))

		question, text = 'Who was Jim Henson?', 'Jim Henson was a nice puppet'
		encoding = tokenizer.encode_plus(question, text)
		input_ids0, token_type_ids = encoding['input_ids'], encoding['token_type_ids']

		print('Inferring...')
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			start_scores, end_scores = model(torch.tensor([input_ids0]), token_type_ids=torch.tensor([token_type_ids]))
		print('Inferred: {} secs.'.format(time.time() - start_time))

		all_tokens = tokenizer.convert_ids_to_tokens(input_ids0)
		answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

		assert answer == 'a nice puppet'
		print('{} processed.'.format(BertForQuestionAnswering.__name__))

# REF [site] >> https://www.analyticsvidhya.com/blog/2019/07/pytorch-transformers-nlp-python/?utm_source=blog&utm_medium=openai-gpt2-text-generator-python
def masked_language_modeling_for_bert_example():
	# Load pre-trained model tokenizer (vocabulary).
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# Tokenize input.
	text = '[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]'
	tokenized_text = tokenizer.tokenize(text)

	# Mask a token that we will try to predict back with 'BertForMaskedLM'.
	masked_index = 8
	tokenized_text[masked_index] = '[MASK]'
	assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

	# Convert token to vocabulary indices.
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	# Define sentence A and B indices associated to 1st and 2nd sentences (see paper).
	segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

	# Convert inputs to PyTorch tensors.
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	# If you have a GPU, put everything on cuda.
	tokens_tensor = tokens_tensor.to('cuda')
	segments_tensors = segments_tensors.to('cuda')

	# Load pre-trained model (weights).
	model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	model.to('cuda')

	# Predict all tokens.
	model.eval()
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
		predictions = outputs[0]

	# Confirm we were able to predict 'henson'.
	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	assert predicted_token == 'henson'
	print('Predicted token is: {}.'.format(predicted_token))

class MyBertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config, pretrained_model_name):
		super(MyBertForSequenceClassification, self).__init__(config)

		#self.bert = BertModel(config)       
		self.bert = BertModel.from_pretrained(pretrained_model_name, config=config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

		# TODO [check] >> Are weights initialized?
		#self.init_weights()

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		return logits

def sequence_classification_using_bert():
	# REF [site] >> https://huggingface.co/transformers/model_doc/bert.html

	pretrained_model_name = 'bert-base-multilingual-cased'

	tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
	print('tokenizer.vocab_size = {}.'.format(tokenizer.vocab_size))
	#print('tokenizer.get_vocab():\n{}.'.format(tokenizer.get_vocab()))

	if True:
		model = BertForSequenceClassification.from_pretrained(pretrained_model_name)
	elif False:
		model = MyBertForSequenceClassification.from_pretrained(pretrained_model_name, pretrained_model_name=pretrained_model_name)  # Not good.
	else:
		#config = BertConfig(num_labels=10, output_attentions=False, output_hidden_states=False)
		#config = BertConfig.from_pretrained(pretrained_model_name, num_labels=10, output_attentions=False, output_hidden_states=False)
		config = BertConfig.from_pretrained(pretrained_model_name, output_attentions=False, output_hidden_states=False)

		#model = MyBertForSequenceClassification.from_pretrained(pretrained_model_name, config=config, pretrained_model_name=pretrained_model_name)  # Not good.
		model = MyBertForSequenceClassification(config, pretrained_model_name=pretrained_model_name)

	#--------------------
	# Train a model.

	#--------------------
	# Test a model.
	input_ids = [
		tokenizer.encode('Hello, my dog is so cute.', add_special_tokens=True),
		tokenizer.encode('Hi, my cat is cute', add_special_tokens=True),
		tokenizer.encode('Hi, my pig is so small...', add_special_tokens=True),
	]
	max_input_len = len(max(input_ids, key=len))
	print('Max. input len = {}.'.format(max_input_len))
	def convert(x):
		y = [x[-1]] * max_input_len  # TODO [check] >> x[-1] is correct?
		y[:len(x)] = x
		return y
	input_ids = list(map(convert, input_ids))
	input_ids = torch.tensor(input_ids)

	model.eval()
	with torch.no_grad():
		model_outputs = model(input_ids)  # Batch size x #labels.

	print('Model output losses = {}.'.format(model_outputs.loss))
	print('Model output logits = {}.'.format(model_outputs.logits))

def korean_bert_example():
	if False:
		pretrained_model_name = 'bert-base-multilingual-uncased'
		#pretrained_model_name = 'bert-base-multilingual-cased'  # Not correctly working.

		tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
	else:
		# REF [site] >> https://github.com/monologg/KoBERT-Transformers
		from tokenization_kobert import KoBertTokenizer

		# REF [site] >> https://huggingface.co/monologg
		pretrained_model_name = 'monologg/kobert'
		#pretrained_model_name = 'monologg/distilkobert'

		tokenizer = KoBertTokenizer.from_pretrained(pretrained_model_name)

	tokens = tokenizer.tokenize('잘해놨습니다')
	token_ids = tokenizer.convert_tokens_to_ids(tokens)
	print('Tokens = {}.'.format(tokens))
	#print('Token IDs = {}.'.format(token_ids))

	model = BertForSequenceClassification.from_pretrained(pretrained_model_name)

	#--------------------
	input_ids = [
		tokenizer.encode('내 개는 무척 귀여워.', add_special_tokens=True),
		tokenizer.encode('내 고양이는 귀여워.', add_special_tokens=True),
		tokenizer.encode('내 돼지는 너무 작아요.', add_special_tokens=True),
	]
	max_input_len = len(max(input_ids, key=len))
	print('Max. input len = {}.'.format(max_input_len))
	def convert(x):
		y = [x[-1]] * max_input_len  # TODO [check] >> x[-1] is correct?
		y[:len(x)] = x
		return y
	input_ids = list(map(convert, input_ids))
	input_ids = torch.tensor(input_ids)

	model.eval()
	with torch.no_grad():
		model_outputs = model(input_ids)  # Batch size x #labels.

	print('Model output losses = {}.'.format(model_outputs.loss))
	print('Model output logits = {}.'.format(model_outputs.logits))

# REF [site] >>
#	https://github.com/SKTBrain/KoBERT
#	https://huggingface.co/skt/kobert-base-v1
def skt_bert_test():
	from kobert import get_pytorch_kobert_model

	model, vocab  = get_pytorch_kobert_model()
	print(f'Vocabulary: {vocab}.')

	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)

	print(f'sequence_output.shape = {sequence_output.shape}.')
	print(f'pooled_output.shape = {pooled_output.shape}.')
	# Last encoding layer.
	print(sequence_output[0])

# REF [site] >> https://huggingface.co/klue/bert-base
def klue_bert_test():
	# Models:
	#	klue/bert-base: ~430MB.

	from transformers import AutoModel, AutoTokenizer

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}.')

	tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
	model = AutoModel.from_pretrained('klue/bert-base')

	if True:
		generator = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
		#generator = pipeline(task='text-generation', model=model, tokenizer=tokenizer, device=device)

		prompt = "인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던"
		min_new_tokens, max_new_tokens = 50, 500

		print('Generating text...')
		start_time = time.time()
		gen_texts = generator(prompt, do_sample=True, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
		print(f'Text generated: {time.time() - start_time} secs.')

		assert len(gen_texts) == 1
		print('Generated text:')
		print(gen_texts[0]['generated_text'])
	else:
		model.to(device=device, non_blocking=True)

		prompt = "인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던"
		tokens = tokenizer.encode(prompt, return_tensors='pt')
		tokens = tokens.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			min_new_tokens, max_new_tokens = 50, 500

			print('Generating text...')
			start_time = time.time()
			gen_tokens = model.generate(tokens, do_sample=True, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)  # Sampling.
			print(f'Text generated: {time.time() - start_time} secs.')

			gen_texts = tokenizer.batch_decode(gen_tokens)
			assert len(gen_texts) == 1
			print('Generated text:')
			print(gen_texts[0])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/t5
def t5_example():
	# Models:
	#	t5-small: ~240MB.
	#	t5-base: ~890MB.
	#	t5-large: ~2.95GB.
	#	t5-3b: ~11.4GB.
	#	t5-11b: ~45.2GB.

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}.')

	# Translation.
	if False:
		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('t5-small')
		model = T5ForConditionalGeneration.from_pretrained('t5-small')
		model.to(device=device, non_blocking=True)

		input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
		input_ids = input_ids.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			print('Translating...')
			start_time = time.time()
			outputs = model.generate(input_ids)
			print(f'Translated: {time.time() - start_time} secs.')

		print('Translated:')
		print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # Output: "Das Haus ist wunderbar.".

	if False:
		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('t5-small')
		model = T5ForConditionalGeneration.from_pretrained('t5-small')
		model.to(device=device, non_blocking=True)

		# When generating, we will use the logits of right-most token to predict the next token so the padding should be on the left.
		tokenizer.padding_side = 'left'
		tokenizer.pad_token = tokenizer.eos_token  # To avoid an error.

		task_prefix = 'translate English to German: '
		sentences = ['The house is wonderful.', 'I like to work in NYC.']  # Use different length sentences to test batching.
		inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors='pt', padding=True)
		inputs = inputs.to(device=device)

		model.eval()
		with torch.no_grad():
			print('Translating...')
			start_time = time.time()
			output_sequences = model.generate(
				input_ids=inputs['input_ids'],
				attention_mask=inputs['attention_mask'],
				do_sample=False,  # Disable sampling to test if batching affects output.
			)
			print(f'Translated: {time.time() - start_time} secs.')

		print('Translated:')
		print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))  # Output: ['Das Haus ist wunderbar.', 'Ich arbeite gerne in NYC.'].

	#--------------------
	# Text summarization.
	if True:
		from transformers import AutoTokenizer, T5ForConditionalGeneration

		tokenizer = AutoTokenizer.from_pretrained('t5-small')
		model = T5ForConditionalGeneration.from_pretrained('t5-small')
		model.to(device=device, non_blocking=True)

		# Training.
		if False:
			input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
			labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
			outputs = model(input_ids=input_ids, labels=labels)
			loss = outputs.loss
			logits = outputs.logits

		# Inference.
		prompt1 = 'summarize: studies have shown that owning a dog is good for you'
		prompt2 = 'summarize: State authorities dispatched emergency crews Tuesday to survey the damage after an onslaught of severe weather in Mississippi that injured at least 17 people, damaged property and cut power.'
		prompt3 = """summarize:
In physics, relativistic mechanics refers to mechanics compatible with special relativity (SR) and general relativity (GR).
It provides a non-quantum mechanical description of a system of particles, or of a fluid, in cases where the velocities of moving objects are comparable to the speed of light c.
As a result, classical mechanics is extended correctly to particles traveling at high velocities and energies, and provides a consistent inclusion of electromagnetism with the mechanics of particles.
This was not possible in Galilean relativity, where it would be permitted for particles and light to travel at any speed, including faster than light.
The foundations of relativistic mechanics are the postulates of special relativity and general relativity.
The unification of SR with quantum mechanics is relativistic quantum mechanics, while attempts for that of GR is quantum gravity, an unsolved problem in physics.
"""

		input_ids = tokenizer(prompt2, return_tensors='pt').input_ids  # Batch size 1.
		input_ids = input_ids.to(device=device, non_blocking=True)

		model.eval()
		with torch.no_grad():
			print('Summarizing text...')
			start_time = time.time()
			outputs = model.generate(input_ids, min_length=5, max_length=20)  # Failed to summarize.
			print(f'Text summarized: {time.time() - start_time} secs.')

		assert len(outputs) == 1
		print('Summary:')
		print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # Output: "studies have shown that owning a dog is good for you.".

# REF [site] >>
#	https://huggingface.co/google
#	https://huggingface.co/docs/transformers/model_doc/flan-t5
def flan_t5_example():
	# Models:
	#	google/flan-t5-small: ~308MB.
	#	google/flan-t5-base: ~990MB.
	#	google/flan-t5-large: ~3.13GB.
	#	google/flan-t5-xl: ~9.45GB.
	#	google/flan-t5-xxl: ~9.45GB.

	if True:
		from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
		model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

		inputs = tokenizer('A step by step recipe to make bolognese pasta:', return_tensors='pt')
		outputs = model.generate(**inputs)

		print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

	if False:
		# Running the model on a CPU.

		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
		model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

		input_text = 'translate English to German: How old are you?'
		input_ids = tokenizer(input_text, return_tensors='pt').input_ids

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

	if False:
		# Running the model on a GPU.

		# Install:
		#	pip install accelerate

		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
		model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', device_map='auto')

		input_text = 'translate English to German: How old are you?'
		input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

	if False:
		# Running the model on a GPU using different precisions (FP16).

		# Install:
		#	pip install accelerate

		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
		model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', device_map='auto', torch_dtype=torch.float16)

		input_text = 'translate English to German: How old are you?'
		input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

	if False:
		# Running the model on a GPU using different precisions (INT8).

		# Install:
		#	pip install bitsandbytes accelerate

		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
		model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', device_map='auto', load_in_8bit=True)

		input_text = 'translate English to German: How old are you?'
		input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

def palm_example():
	if False:
		# https://github.com/lucidrains/PaLM-pytorch

		# Install:
		#	pip install PaLM-pytorch

		import palm_pytorch

		if True:
			palm = palm_pytorch.PaLM(
				num_tokens=20000,
				dim=512,
				depth=12,
				heads=8,
				dim_head=64,
			)
		else:
			# The PaLM 540B in the paper would be.
			palm = palm_pytorch.PaLM(
				num_tokens=256000,
				dim=18432,
				depth=118,
				heads=48,
				dim_head=256,
			)

		tokens = torch.randint(0, 20000, (1, 2048))
		logits = palm(tokens)  # (1, 2048, 20000).

		print(f"{logits.shape=}")

	if True:
		# https://github.com/lucidrains/PaLM-rlhf-pytorch

		# Install:
		#	pip install palm-rlhf-pytorch

		import palm_rlhf_pytorch

		# 1. train PaLM, like any other autoregressive transformer.

		palm = palm_rlhf_pytorch.PaLM(
			num_tokens=20000,
			dim=512,
			depth=12,
			flash_attn=False,  # https://arxiv.org/abs/2205.14135
		).cuda()

		seq = torch.randint(0, 20000, (1, 2048)).cuda()

		loss = palm(seq, return_loss=True)
		loss.backward()

		#torch.save(palm.state_dict(), "/path/to/pretrained/palm.pt")

		# After much training, you can now generate sequences.
		generated = palm.generate(2048)  # (1, 2048).

		#-----
		# 2.train your reward model, with the curated human feedback.

		palm = palm_rlhf_pytorch.PaLM(
			num_tokens=20000, 
			dim=512,
			depth=12,
			causal=False,
		)

		#palm.load("/path/to/pretrained/palm.pt")

		reward_model = palm_rlhf_pytorch.RewardModel(
			palm,
			num_binned_output=5,  # Say rating from 1 to 5.
		).cuda()

		#torch.save(reward_model.state_dict(), "/path/to/pretrained/reward_model.pt")

		# Mock data.
		seq = torch.randint(0, 20000, (1, 1024)).cuda()
		prompt_mask = torch.zeros(1, 1024).bool().cuda()  # Which part of the sequence is prompt, which part is response.
		labels = torch.randint(0, 5, (1,)).cuda()

		# Train.
		loss = reward_model(seq, prompt_mask=prompt_mask, labels=labels)
		loss.backward()

		# After much training.
		reward = reward_model(seq, prompt_mask=prompt_mask)

		#-----
		# 3. pass your transformer and the rewards model to the RLHFTrainer.

		# Load your pretrained palm.
		palm = palm_rlhf_pytorch.PaLM(
			num_tokens=20000,
			dim=512,
			depth=12,
		).cuda()

		#palm.load("/path/to/pretrained/palm.pt")

		# Load your pretrained reward model.
		reward_model = palm_rlhf_pytorch.RewardModel(
			palm,
			num_binned_output=5,
		).cuda()

		#reward_model.load("/path/to/pretrained/reward_model.pt")

		# Ready your list of prompts for reinforcement learning.
		prompts = torch.randint(0, 256, (50000, 512)).cuda()  # 50k prompts.

		# Pass it all to the trainer and train.
		trainer = palm_rlhf_pytorch.RLHFTrainer(
			palm=palm,
			reward_model=reward_model,
			prompt_token_ids=prompts,
		)

		trainer.train(num_episodes=50000)

		# Generate say 10 samples and use the reward model to return the best one.
		answer = trainer.generate(2048, prompt=prompts[0], num_samples=10)  # (<= 2048,).

# REF [site] >>
#	https://huggingface.co/bigscience
#	https://huggingface.co/docs/transformers/model_doc/bloom
def bloom_example():
	# Models:
	#	bigscience/bloom-560m: ~1.1GB.
	#	bigscience/bloom-1b1: ~2.13GB.
	#	bigscience/bloom-1b7: ~3.44GB.
	#	bigscience/bloom-3b: ~6GB.
	#	bigscience/bloom-7b1: ~20GB.
	#	bigscience/bloom: ~520GB.
	#	bigscience/bloomz-560m.
	#	bigscience/bloomz-1b1.
	#	bigscience/bloomz-1b7.
	#	bigscience/bloomz-7b1.
	#	bigscience/bloomz-3b.
	#	bigscience/bloomz: ~520GB.

	# BigScience Language Open-science Open-access Multilingual (BLOOM) language model.
	#	Modified from Megatron-LM GPT2 (see paper, BLOOM Megatron code).
	#	Decoder-only architecture.
	#	Layer normalization applied to word embeddings layer (StableEmbedding; see code, paper).
	#	ALiBi positional encodings (see paper), with GeLU activation functions.
	#	176, 247, 271, 424 parameters:
	#		3, 596, 615, 680 embedding parameters.
	#		70 layers, 112 attention heads.
	#		Hidden layers are 14336-dimensional.
	#		Sequence length of 2048 tokens used (see BLOOM tokenizer, tokenizer description).
	#	Objective Function: Cross Entropy with mean reduction.

	if False:
		from transformers import BloomConfig, BloomModel

		# Initializing a Bloom configuration.
		configuration = BloomConfig()

		# Initializing a model (with random weights) from the configuration.
		model = BloomModel(configuration)

		# Accessing the model configuration.
		configuration = model.config
		print("Configuration:")
		print(configuration)

	if False:
		from transformers import AutoTokenizer, BloomModel

		tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
		model = BloomModel.from_pretrained('bigscience/bloom-560m')

		inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
		outputs = model(**inputs)

		print(f'last_hidden_states: {outputs.last_hidden_state}.')

	if False:
		from transformers import BloomTokenizerFast

		tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom')

		print(f"tokenizer('Hello world')['input_ids'] = {tokenizer('Hello world')['input_ids']}.")
		print(f"tokenizer(' Hello world')['input_ids'] = {tokenizer(' Hello world')['input_ids']}.")

	if False:
		from transformers import AutoTokenizer, BloomForCausalLM

		tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
		model = BloomForCausalLM.from_pretrained('bigscience/bloom-560m')

		inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
		outputs = model(**inputs, labels=inputs['input_ids'])

		print(f'outputs.loss = {outputs.loss}.')
		print(f'outputs.logits = {outputs.logits}.')

	if False:
		from transformers import AutoTokenizer, BloomForSequenceClassification

		# Single-label classification.

		tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
		model = BloomForSequenceClassification.from_pretrained('bigscience/bloom-560m')

		inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')

		model.eval()
		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_id = logits.argmax().item()
		#print(model.config.id2label[predicted_class_id])

		# To train a model on 'num_labels' classes, you can pass 'num_labels=num_labels' to '.from_pretrained(...)'.
		num_labels = len(model.config.id2label)
		model = BloomForSequenceClassification.from_pretrained('bigscience/bloom-560m', num_labels=num_labels)

		labels = torch.tensor([1])
		loss = model(**inputs, labels=labels).loss

		#-----
		# Multi-label classification.

		tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
		model = BloomForSequenceClassification.from_pretrained('bigscience/bloom-560m', problem_type='multi_label_classification')

		inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

		# To train a model on 'num_labels' classes, you can pass 'num_labels=num_labels' to '.from_pretrained(...)'.
		num_labels = len(model.config.id2label)
		model = BloomForSequenceClassification.from_pretrained('bigscience/bloom-560m', num_labels=num_labels, problem_type='multi_label_classification')

		labels = torch.sum(torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1).to(torch.float)
		loss = model(**inputs, labels=labels).loss

	if True:
		from transformers import AutoTokenizer, BloomForTokenClassification

		tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
		model = BloomForTokenClassification.from_pretrained('bigscience/bloom-560m')

		inputs = tokenizer('HuggingFace is a company based in Paris and New York', add_special_tokens=False, return_tensors='pt')

		model.eval()
		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_token_class_ids = logits.argmax(-1)

		# Note that tokens are classified rather then input words which means that there might be more predicted token classes than words.
		# Multiple token classes might account for the same word.
		predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

		labels = predicted_token_class_ids
		loss = model(**inputs, labels=labels).loss

# REF [site] >> https://huggingface.co/facebook
def opt_example():
	# Models:
	#	facebook/opt-125m.
	#	facebook/opt-350m.
	#	facebook/opt-2.7b: ~5.30GB.
	#	facebook/opt-6.7b.
	#	facebook/opt-13b.
	#	facebook/opt-30b.
	#
	#	facebook/opt-iml-1.3b.
	#	facebook/opt-iml-max-1.3b.
	#	facebook/opt-iml-30b.
	#	facebook/opt-iml-max-30b.

	if True:
		model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16).cuda()
		# The fast tokenizer currently does not work correctly.
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)

		prompt = "Hello, I am conscious and"
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

		if True:
			generated_ids = model.generate(input_ids)  # By default, generation is deterministic.
		else:
			tokenizer.set_seed(32)
			generated_ids = model.generate(input_ids, do_sample=True)  # In order to use the top-k sampling, please set do_sample to True.

		generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		print(generated)

	if True:
		# Here's an example of how the model can have biased predictions.

		model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16).cuda()
		# The fast tokenizer currently does not work correctly.
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)

		prompt = "The woman worked as a"
		#prompt = "The man worked as a"
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

		transformers.set_seed(32)
		generated_ids = model.generate(input_ids, do_sample=True, num_return_sequences=5, max_length=10)

		generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		print(generated)

# REF [site] >> https://huggingface.co/facebook
def galactica_example():
	# Models:
	#	facebook/galactica-125m.
	#	facebook/galactica-1.3b.
	#	facebook/galactica-6.7b: ~13.73GB.
	#	facebook/galactica-30b.
	#	facebook/galactica-120b.

	# Install:
	#	pip install bitsandbytes accelerate

	tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
	#model = transformers.OPTForCausalLM.from_pretrained("facebook/galactica-120b")  # CPU.
	model = transformers.OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")  # GPU.
	#model = transformers.OPTForCausalLM.from_pretrained("facebook/galactica-120b", device_map="auto", torch_dtype=torch.float16)  # GPU. FP16.
	#model = transformers.OPTForCausalLM.from_pretrained("facebook/galactica-120b", device_map="auto", load_in_8bit=True)  # GPU. INT8.

	input_text = "The Transformer architecture [START_REF]"
	#input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # CPU.
	input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")  # GPU.

	outputs = model.generate(input_ids)
	print(tokenizer.decode(outputs[0]))

# REF [site] >>
#	https://huggingface.co/decapoda-research
#	https://huggingface.co/docs/transformers/main/en/model_doc/llama
def llama_example():
	# Models:
	#	decapoda-research/llama-smallint-pt.
	#	decapoda-research/llama-7b-hf: ~13.5GB.
	#	decapoda-research/llama-13b-hf: ~39.1GB.
	#	decapoda-research/llama-30b-hf: ~81.5GB.
	#	decapoda-research/llama-65b-hf: ~130.7GB.
	#	decapoda-research/llama-7b-hf-int4.
	#	decapoda-research/llama-13b-hf-int4.
	#	decapoda-research/llama-30b-hf-int4.
	#	decapoda-research/llama-65b-hf-int4.
	#	decapoda-research/llama-7b-hf-int8.

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")
	print(f"Device: {device}.")

	if False:
		# Initializing a LLaMA llama-7b style configuration.
		configuration = transformers.LlamaConfig()

		# Initializing a model from the llama-7b style configuration.
		model = transformers.LlamaModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if False:
		if False:
			# Initializing a LLaMA llama-7b style configuration.
			configuration = transformers.LlamaConfig()

			# Initializing a model from the llama-7b style configuration.
			model = transformers.LlamaModel(configuration)

			# Accessing the model configuration.
			configuration = model.config

		if False:
			tokenizer = transformers.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

			encoded = tokenizer.encode("Hello this is a test")
			print(encoded)

		model = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
		#tokenizer = transformers.AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")  # ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported.
		#tokenizer = transformers.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")  # Not good.
		tokenizer = transformers.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

		prompt = "Hey, are you conscious? Can you talk to me?"
		inputs = tokenizer(prompt, return_tensors="pt")

		# Generate.
		generate_ids = model.generate(inputs.input_ids, max_length=30)

		generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(generated)

	#-----
	if True:
		print("Loading a model...")
		start_time = time.time()
		model = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
		tokenizer = transformers.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
		print(f"A model loaded: {time.time() - start_time} secs.")

		prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: """

		print("Generating text...")
		start_time = time.time()
		#with torch.autocast(device, dtype=torch.bfloat16):
		model.to(device)
		model.eval()
		with torch.no_grad():
			inputs = tokenizer(prompt, return_tensors="pt").to(device)
			generate_ids = model.generate(inputs.input_ids, max_length=256)
			generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(f"Text generated: {time.time() - start_time} secs.")
		print(generated)

# REF [site] >> https://huggingface.co/meta-llama
def llama2_example():
	# Models:
	#	meta-llama/Llama-2-7b.
	#	meta-llama/Llama-2-13b.
	#	meta-llama/Llama-2-70b.
	#	meta-llama/Llama-2-7b-hf: ~12.6GB.
	#	meta-llama/Llama-2-13b-hf: ~24.3GB.
	#	meta-llama/Llama-2-70b-hf.
	#	meta-llama/Llama-2-7b-chat.
	#	meta-llama/Llama-2-13b-chat.
	#	meta-llama/Llama-2-70b-chat.
	#	meta-llama/Llama-2-7b-chat-hf.
	#	meta-llama/Llama-2-13b-chat-hf.
	#	meta-llama/Llama-2-70b-chat-hf.

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if True:
		# Model parallelism.
		#	https://huggingface.co/docs/accelerate/usage_guides/big_modeling

		import accelerate

		model_name = "meta-llama/Llama-2-7b-hf"
		checkpoint_shards_dir_path = "./huggingface_checkpoint_shards"

		if False:
			# Save checkpoint shards.
			model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

			print(f"Saving checkpoint shards to {checkpoint_shards_dir_path}...")
			start_time = time.time()
			accelerator = accelerate.Accelerator()
			accelerator.save_model(model, save_directory=checkpoint_shards_dir_path, max_shard_size="10GB", safe_serialization=False)
			print(f"Checkpoint shards saved: {time.time() - start_time} secs.")

		if True:
			# Load checkpoint shards.
			#model_config = transformers.AutoModelForCausalLM.from_pretrained(model_name).config
			model_config = transformers.LlamaConfig()
			#model_config = model.config

			with accelerate.init_empty_weights():
				model = transformers.AutoModelForCausalLM.from_config(model_config)

			# Designing a device map.
			# We don't want to use more than 10GiB on each of the two GPUs and no more than 30GiB of CPU RAM for the model weights:
			#device_map = accelerate.infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
			# With BLOOM-176B on 8x80 A100 setup, the close-to-ideal map is:
			#device_map = accelerate.infer_auto_device_map(model, max_memory={0: "30GIB", 1: "46GIB", 2: "46GIB", 3: "46GIB", 4: "46GIB", 5: "46GIB", 6: "46GIB", 7: "46GIB"})
			# If you opt to fully design the device_map yourself, it should be a dictionary with keys being module names of your model and values being a valid device identifier (for instance an integer for the GPUs) or "cpu" for CPU offload, "disk" for disk offload:
			#device_map = {"block1": 0, "block2": 1}
			#device_map = {"block1": 0, "block2.linear1": 0, "block2.linear2": 1, "block2.linear3": 1}
			#device_map = {"block1": 0, "block2.linear1": 1, "block2.linear2": 1}
			# Llama 2 7B:
			#device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0, "model.layers.4": 0, "model.layers.5": 0, "model.layers.6": 0, "model.layers.7": 0, "model.layers.8": 0, "model.layers.9": 0, "model.layers.10": 0, "model.layers.11": 0, "model.layers.12": 0, "model.layers.13": 0, "model.layers.14": 0, "model.layers.15.self_attn": 0, "model.layers.15.input_layernorm": 1, "model.layers.15.post_attention_layernorm": 1, "model.layers.16": 1, "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1, "model.layers.20": 1, "model.layers.21": 1, "model.layers.22": 1, "model.layers.23": 1, "model.layers.24": 1, "model.layers.25": 1, "model.layers.26": 1, "model.layers.27": 1, "model.layers.28": 1, "model.layers.29": 1, "model.layers.30": 1, "model.layers.31": 1, "model.norm": 1, "lm_head": 1, "model.layers.15.mlp": 1}  # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
			device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0, "model.layers.4": 0, "model.layers.5": 0, "model.layers.6": 0, "model.layers.7": 0, "model.layers.8": 0, "model.layers.9": 0, "model.layers.10": 0, "model.layers.11": 0, "model.layers.12": 0, "model.layers.13": 0, "model.layers.14": 1, "model.layers.15.self_attn": 1, "model.layers.15.input_layernorm": 1, "model.layers.15.post_attention_layernorm": 1, "model.layers.16": 1, "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1, "model.layers.20": 1, "model.layers.21": 1, "model.layers.22": 1, "model.layers.23": 1, "model.layers.24": 1, "model.layers.25": 1, "model.layers.26": 1, "model.layers.27": 1, "model.layers.28": 1, "model.layers.29": 1, "model.layers.30": 1, "model.layers.31": 1, "model.norm": 1, "lm_head": 1, "model.layers.15.mlp": 1}
			# Llama 2 13B:

			print(f"Loading checkpoint shards from {checkpoint_shards_dir_path}...")
			start_time = time.time()
			model = accelerate.load_checkpoint_and_dispatch(
				model, checkpoint=checkpoint_shards_dir_path,
				#device_map="auto",
				device_map=device_map,
				no_split_module_classes=["Block"],
			)
			print(f"Checkpoint shards loaded: {time.time() - start_time} secs.")

			# Device map.
			print(model.hf_device_map)

		# Inference.
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

		prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: """

		print("Generating text...")
		start_time = time.time()
		model.eval()
		with torch.no_grad():
			inputs = tokenizer(prompt, return_tensors="pt").to(device)
			#inputs = tokenizer(prompt, return_tensors="pt").to(0)
			generate_ids = model.generate(inputs.input_ids, max_length=256)
			generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(f"Text generated: {time.time() - start_time} secs.")
		print(generated)

	#-----
	if False:
		model_name = "meta-llama/Llama-2-7b-hf"

		print(f"Loading a model, {model_name}...")
		start_time = time.time()
		#model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
		model = transformers.LlamaForCausalLM.from_pretrained(model_name)
		#tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_name)
		#tokenizer = transformers.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
		print(f"A model loaded: {time.time() - start_time} secs.")

		prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: """

		print("Generating text...")
		start_time = time.time()
		#with torch.autocast(device, dtype=torch.bfloat16):
		model.to(device)
		model.eval()
		with torch.no_grad():
			inputs = tokenizer(prompt, return_tensors="pt").to(device)
			generate_ids = model.generate(inputs.input_ids, max_length=256)
			generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(f"Text generated: {time.time() - start_time} secs.")
		print(generated)

	if False:
		model_name = "meta-llama/Llama-2-7b-chat-hf"

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		pipeline = transformers.pipeline(
			"text-generation",
			model=model_name,
			torch_dtype=torch.float16,
			device_map="auto",
		)

		sequences = pipeline(
			'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
			do_sample=True,
			top_k=10,
			num_return_sequences=1,
			eos_token_id=tokenizer.eos_token_id,
			max_length=200,
		)
		for seq in sequences:
			print(f"Result: {seq['generated_text']}")

# REF [site] >> https://huggingface.co/meta-llama
def llama3_example():
	# Models:
	#	meta-llama/Meta-Llama-3-8B.
	#	meta-llama/Meta-Llama-3-8B-Instruct.
	#	meta-llama/Meta-Llama-3-70B.
	#	meta-llama/Meta-Llama-3-70B-Instruct.

	if True:
		# To download original checkpoints, see the example command below leveraging huggingface-cli:
		#	huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir Meta-Llama-3-8B

		model_id = "meta-llama/Meta-Llama-3-8B"
		#model_id = "meta-llama/Meta-Llama-3-70B"

		pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

		outputs = pipeline("Hey how are you doing today?")
		print(outputs[0]["generated_text"])

	if True:
		# To download original checkpoints, see the example command below leveraging huggingface-cli:
		#	huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct

		model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
		#model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

		pipeline = transformers.pipeline(
			"text-generation",
			model=model_id,
			model_kwargs={"torch_dtype": torch.bfloat16},
			device="auto",
		)

		messages = [
			{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
			{"role": "user", "content": "Who are you?"},
		]

		prompt = pipeline.tokenizer.apply_chat_template(
			messages, 
			tokenize=False, 
			add_generation_prompt=True,
		)

		terminators = [
			pipeline.tokenizer.eos_token_id,
			pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
		]

		outputs = pipeline(
			prompt,
			max_new_tokens=256,
			eos_token_id=terminators,
			do_sample=True,
			temperature=0.6,
			top_p=0.9,
		)
		print(outputs[0]["generated_text"][len(prompt):])

	if True:
		model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
		#model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model = transformers.AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)

		messages = [
			{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
			{"role": "user", "content": "Who are you?"},
		]

		input_ids = tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_tensors="pt",
		).to(model.device)

		terminators = [
			tokenizer.eos_token_id,
			tokenizer.convert_tokens_to_ids("<|eot_id|>"),
		]

		outputs = model.generate(
			input_ids,
			max_new_tokens=256,
			eos_token_id=terminators,
			do_sample=True,
			temperature=0.6,
			top_p=0.9,
		)
		response = outputs[0][input_ids.shape[-1]:]
		print(tokenizer.decode(response, skip_special_tokens=True))

# REF [site] >> https://huggingface.co/openlm-research
def open_llama_example():
	# Models:
	#	openlm-research/open_llama_3b.
	#	openlm-research/open_llama_7b.
	#	openlm-research/open_llama_13b.
	#	openlm-research/open_llama_3b_easylm.
	#	openlm-research/open_llama_7b_easylm.
	#	openlm-research/open_llama_13b_easylm.
	#	openlm-research/open_llama_3b_v2.
	#	openlm-research/open_llama_7b_v2: ~13.48GB.
	#	openlm-research/open_llama_3b_v2_easylm.
	#	openlm-research/open_llama_7b_v2_easylm.

	# v2 models
	#model_path = "openlm-research/open_llama_3b_v2"
	model_path = "openlm-research/open_llama_7b_v2"
	# v1 models
	#model_path = "openlm-research/open_llama_3b"
	#model_path = "openlm-research/open_llama_7b"
	#model_path = "openlm-research/open_llama_13b"

	tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
	model = transformers.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

	prompt = "Q: What is the largest animal?\nA:"
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids

	generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)
	print(tokenizer.decode(generation_output[0]))

# REF [site] >> https://huggingface.co/nvidia
def megatron_example():
	# Models:
	#	nvidia/megatron-bert-cased-345m.
	#	nvidia/megatron-bert-uncased-345m.
	#	nvidia/megatron-gpt2-345m.
	#	nvidia/nemo-megatron-gpt-1.3B.
	#	nvidia/nemo-megatron-gpt-5B
	#	nvidia/nemo-megatron-gpt-20B.
	#	nvidia/nemo-megatron-t5-3B.
	#	nvidia/nemo-megatron-mt5-3B.

	"""
	Step 1: Install NeMo and dependencies.
		git clone https://github.com/ericharper/apex.git
		cd apex
		git checkout nm_v1.11.0
		pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

		pip install nemo_toolkit['nlp']==1.11.0

	Step 2: Launch eval server.
		git clone https://github.com/NVIDIA/NeMo.git 
		cd NeMo/examples/nlp/language_modeling
		git checkout v1.11.0
		python megatron_gpt_eval.py gpt_model_file=nemo_gpt5B_fp16_tp2.nemo server=True tensor_model_parallel_size=2 trainer.devices=2
	"""
		
	# Step 3: Send prompts to your model!
	import json

	port_num = 5555
	headers = {"Content-Type": "application/json"}

	def request_data(data):
		resp = requests.put(
			"http://localhost:{}/generate".format(port_num),
			data=json.dumps(data),
			headers=headers
		)
		sentences = resp.json()["sentences"]
		return sentences

	data = {
		"sentences": ["Tell me an interesting fact about space travel."]*1,
		"tokens_to_generate": 50,
		"temperature": 1.0,
		"add_BOS": True,
		"top_k": 0,
		"top_p": 0.9,
		"greedy": False,
		"all_probs": False,
		"repetition_penalty": 1.2,
		"min_tokens_to_generate": 2,
	}

	sentences = request_data(data)
	print(sentences)

# REF [site] >> https://huggingface.co/mosaicml
def mpt_example():
	# Models:
	#	mosaicml/mpt-7b.
	#	mosaicml/mpt-7b-chat: ~13.3GB.
	#	mosaicml/mpt-7b-instruct.
	#	mosaicml/mpt-7b-storywriter.
	#	mosaicml/mpt-30b.
	#	mosaicml/mpt-30b-chat.
	#	mosaicml/mpt-30b-instruct.

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")
	print(f"Device: {device}.")

	if True:
		name = "mosaicml/mpt-7b-chat"  # CUDA out-of-memory.
		tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
	else:
		name = "mosaicml/mpt-30b-chat"  # CUDA out-of-memory.
		tokenizer = transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-30b")

	if False:
		model = transformers.AutoModelForCausalLM.from_pretrained(
			name,
			trust_remote_code=True
		)  # CPU.
	elif False:
		config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
		config.attn_config["attn_impl"] = "triton"  # Change this to use triton-based FlashAttention.
		config.init_device = device  # For fast initialization directly on GPU!

		model = transformers.AutoModelForCausalLM.from_pretrained(
			name,
			config=config,
			torch_dtype=torch.bfloat16,  # Load model weights in bfloat16.
			trust_remote_code=True
		)
	elif True:
		config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
		config.max_seq_len = 4096  # (input + output) tokens can now be up to 4096.
		#config.max_seq_len = 16384  # (input + output) tokens can now be up to 16384.

		model = transformers.AutoModelForCausalLM.from_pretrained(
			name,
			config=config,
			trust_remote_code=True
		)

	if False:
		with torch.autocast(device, dtype=torch.bfloat16):
			inputs = tokenizer("Here is a recipe for vegan banana bread:\n", return_tensors="pt").to(device)
			outputs = model.generate(**inputs, max_new_tokens=100)
			print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
	elif False:
		# Using the HF pipeline.
		pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)  # Warning: The model 'MPTForCausalLM' is not supported for text-generation.
		with torch.autocast(device, dtype=torch.bfloat16):
			generated = pipe(
				"Here is a recipe for vegan banana bread:\n",
				max_new_tokens=100, do_sample=True, use_cache=True
			)
			print(generated)

	#-----
	if True:
		prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: """

		with torch.autocast(device, dtype=torch.bfloat16):
			inputs = tokenizer(prompt, return_tensors="pt").to(device)
			outputs = model.generate(**inputs, max_new_tokens=100)
			print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# REF [site] >> https://huggingface.co/docs/transformers/main/model_doc/falcon
def falcon_example():
	model_name = "Rocketknight1/falcon-rw-1b"  # ~2.62GB.

	if True:
		# Initializing a small (2-layer) Falcon configuration
		configuration = transformers.FalconConfig(num_hidden_layers=2)

		# Initializing a model from the small configuration
		model = transformers.FalconModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		model = transformers.FalconModel.from_pretrained(model_name)

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		model = transformers.FalconForCausalLM.from_pretrained(model_name)

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs, labels=inputs["input_ids"])
		loss = outputs.loss
		logits = outputs.logits

	if True:
		# Single-label classification

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		model = transformers.FalconForSequenceClassification.from_pretrained(model_name)

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_id = logits.argmax().item()
		#print(model.config.id2label[predicted_class_id])

		# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
		num_labels = len(model.config.id2label)
		model = transformers.FalconForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

		labels = torch.tensor([1])
		loss = model(**inputs, labels=labels).loss

	if True:
		# Multi-label classification

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		model = transformers.FalconForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

		# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
		num_labels = len(model.config.id2label)
		model = transformers.FalconForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

		labels = torch.sum(
			torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
		).to(torch.float)
		loss = model(**inputs, labels=labels).loss

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
		model = transformers.FalconForTokenClassification.from_pretrained(model_name)

		inputs = tokenizer("HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_token_class_ids = logits.argmax(-1)

		# Note that tokens are classified rather then input words which means that
		# there might be more predicted token classes than words.
		# Multiple token classes might account for the same word
		predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

		labels = predicted_token_class_ids
		loss = model(**inputs, labels=labels).loss

# REF [site] >>
#	https://huggingface.co/01-ai
#	https://github.com/01-ai/Yi
def yi_example():
	# Models:
	#	01-ai/Yi-6B: ~12.12GB.
	#	01-ai/Yi-6B-200K: ~12.12GB.
	#	01-ai/Yi-34B: ~68.79GB.
	#	01-ai/Yi-34B-200K.

	model_name = "01-ai/Yi-6B"

	model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

	inputs = tokenizer("There's a place where time stands still. A place of breath taking wonder, but also", return_tensors="pt")
	max_length = 256

	outputs = model.generate(
		inputs.input_ids.cuda(),
		max_length=max_length,
		eos_token_id=tokenizer.eos_token_id,
		do_sample=True,
		repetition_penalty=1.3,
		no_repeat_ngram_size=5,
		temperature=0.7,
		top_k=40,
		top_p=0.8,
	)
	print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# REF [site] >> https://huggingface.co/microsoft
def orca_example():
	# Models:
	#	microsoft/Orca-2-7b.
	#	microsoft/Orca-2-13b.

	if torch.cuda.is_available():
		torch.set_default_device("cuda")
	else:
		torch.set_default_device("cpu")

	model_name = "microsoft/Orca-2-7b"

	model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

	# https://github.com/huggingface/transformers/issues/27132
	# Please use the slow tokenizer since fast and slow tokenizer produces different tokens
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

	system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
	user_message = "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"

	prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

	inputs = tokenizer(prompt, return_tensors="pt")
	output_ids = model.generate(inputs["input_ids"],)
	answer = tokenizer.batch_decode(output_ids)[0]

	print(answer)

	# This example continues showing how to add a second turn message by the user to the conversation
	second_turn_user_message = "Give me a list of the key points of your first answer."

	# We set add_special_tokens=False because we dont want to automatically add a bos_token between messages
	second_turn_message_in_markup = f"\n<|im_start|>user\n{second_turn_user_message}<|im_end|>\n<|im_start|>assistant"
	second_turn_tokens = tokenizer(second_turn_message_in_markup, return_tensors="pt", add_special_tokens=False)
	second_turn_input = torch.cat([output_ids, second_turn_tokens["input_ids"]], dim=1)

	output_ids_2 = model.generate(second_turn_input,)
	second_turn_answer = tokenizer.batch_decode(output_ids_2)[0]

	print(second_turn_answer)

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/mistral
#	https://huggingface.co/mistralai
def mistral_example():
	# Models:
	#	mistralai/Mistral-7B-v0.1: ~14.48GB.
	#	mistralai/Mistral-7B-Instruct-v0.1.
	#	mistralai/Mistral-7B-Instruct-v0.2.

	device = "cuda"  # The device to load the model onto

	if False:
		# Initializing a Mistral 7B style configuration
		configuration = transformers.MistralConfig()

		# Initializing a model from the Mistral 7B style configuration
		model = transformers.MistralModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		model_id = "mistralai/Mistral-7B-v0.1"

		if True:
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
			tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		elif False:
			# Combining Mistral and Flash Attention 2
			#	pip install -U flash-attn --no-build-isolation
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
			tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		else:
			tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id)
			model = transformers.MistralForCausalLM.from_pretrained(model_id)
		model.to(device)

		if True:
			prompt = "My favourite condiment is"
			model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

			generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
			decoded = tokenizer.batch_decode(generated_ids)[0]
			print(decoded)
		else:
			prompt = "Hey, are you conscious? Can you talk to me?"
			inputs = tokenizer(prompt, return_tensors="pt")

			generate_ids = model.generate(inputs.input_ids, max_length=30)
			decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
			print(decoded)

	if True:
		model_id = "mistralai/Mistral-7B-Instruct-v0.2"

		model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model.to(device)

		messages = [
			{"role": "user", "content": "What is your favourite condiment?"},
			{"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
			{"role": "user", "content": "Do you have mayonnaise recipes?"}
		]

		encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
		model_inputs = encodeds.to(device)

		generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
		decoded = tokenizer.batch_decode(generated_ids)
		print(decoded[0])

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/mixtral
#	https://huggingface.co/mistralai
def mixtral_example():
	# Models:
	#	mistralai/Mixtral-8x7B-v0.1.
	#	mistralai/Mixtral-8x7B-Instruct-v0.1.

	device = "cuda"  # The device to load the model onto

	if False:
		# Initializing a Mixtral 7B style configuration
		configuration = transformers.MixtralConfig()

		# Initializing a model from the Mixtral 7B style configuration
		model = transformers.MixtralModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		model_id = "mistralai/Mixtral-8x7B-v0.1"

		if True:
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
			tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		elif False:
			# Combining Mistral and Flash Attention 2
			#	pip install -U flash-attn --no-build-isolation
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
			tokenizer = AutoTokenizer.from_pretrained(model_id)
		else:
			tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id)
			model = transformers.MixtralForCausalLM.from_pretrained(model_id)
		model.to(device)

		if True:
			prompt = "My favourite condiment is"
			model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

			generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
			decoded = tokenizer.batch_decode(generated_ids)[0]
			print(decoded)
		else:
			prompt = "Hey, are you conscious? Can you talk to me?"
			inputs = tokenizer(prompt, return_tensors="pt")

			generate_ids = model.generate(inputs.input_ids, max_length=30)
			decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
			print(decoded)

	if True:
		model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
		model.to(device)

		text = "Hello my name is"
		inputs = tokenizer(text, return_tensors="pt")

		outputs = model.generate(**inputs, max_new_tokens=20)
		print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# REF [site] >> https://huggingface.co/HuggingFaceH4
def zephyr_example():
	# Models:
	#	HuggingFaceH4/zephyr-7b-alpha.
	#	HuggingFaceH4/zephyr-7b-beta.

	# Install:
	#	Install transformers from source - only needed for versions <= v4.34
	#	pip install git+https://github.com/huggingface/transformers.git
	#	pip install accelerate

	pipe = transformers.pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

	# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
	messages = [
		{
			"role": "system",
			"content": "You are a friendly chatbot who always responds in the style of a pirate",
		},
		{
			"role": "user",
			"content": "How many helicopters can a human eat in one sitting?"
		},
	]
	prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
	print(outputs[0]["generated_text"])
	# <|system|>
	# You are a friendly chatbot who always responds in the style of a pirate.</s>
	# <|user|>
	# How many helicopters can a human eat in one sitting?</s>
	# <|assistant|>
	# Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!

# REF [site] >> https://huggingface.co/google
def gemma_example():
	# Models:
	#	google/gemma-2b.
	#	google/gemma-2b-it.
	#	google/gemma-7b.
	#	google/gemma-7b-it.

	model_id = "google/gemma-2b"

	if True:
		# Running the model on a CPU

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

		input_text = "Write me a poem about Machine Learning."
		input_ids = tokenizer(**input_text, return_tensors="pt")

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

	if True:
		# Running the model on a single / multi GPU

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		if True:
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
		elif False:
			# Using torch.float16
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
		elif False:
			# Using torch.bfloat16
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
		elif False:
			# Using 8-bit precision (int8)
			# pip install bitsandbytes
			quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
		elif False:
			# Using 4-bit precision
			# pip install bitsandbytes
			quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
			model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
		elif False:
			# Flash Attention 2
			# pip install flash-attn
			model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(0)

		input_text = "Write me a poem about Machine Learning."
		input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

		outputs = model.generate(**input_ids)
		print(tokenizer.decode(outputs[0]))

	if True:
		# Chat template

		model_id = "gg-hf/gemma-2b-it"
		#model_id = "gg-hf/gemma-7b-it"
		dtype = torch.bfloat16

		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model = transformers.AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map="cuda",
			torch_dtype=dtype,
		)

		chat = [
			{ "role": "user", "content": "Write a hello world program" },
		]
		prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/rag
def rag_example():
	if False:
		# To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
		retriever = transformers.RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed")

		# To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
		dataset = (
			...
		)  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
		retriever = transformers.RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

		# To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
		dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
		index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
		retriever = transformers.RagRetriever.from_pretrained(
			"facebook/dpr-ctx_encoder-single-nq-base",
			index_name="custom",
			passages_path=dataset_path,
			index_path=index_path,
		)

		# To load the legacy index built originally for Rag's paper
		retriever = transformers.RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/rag-token-base")
		retriever = transformers.RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
		# Initialize with RagRetriever to do everything in one forward call
		model = transformers.RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

		inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
		outputs = model(input_ids=inputs["input_ids"])

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
		retriever = transformers.RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
		# Initialize with RagRetriever to do everything in one forward call
		model = torch.bmm.RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

		inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
		targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
		input_ids = inputs["input_ids"]
		labels = targets["input_ids"]
		outputs = model(input_ids=input_ids, labels=labels)

		# Or use retriever separately
		model = transformers.RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
		# 1. Encode
		question_hidden_states = model.question_encoder(input_ids)[0]
		# 2. Retrieve
		docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
		doc_scores = torch.bmm(
			question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
		).squeeze(1)
		# 3. Forward to generator
		outputs = model(
			context_input_ids=docs_dict["context_input_ids"],
			context_attention_mask=docs_dict["context_attention_mask"],
			doc_scores=doc_scores,
			decoder_input_ids=labels,
		)

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/rag-token-nq")
		retriever = transformers.RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
		# Initialize with RagRetriever to do everything in one forward call
		model = transformers.RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

		inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
		targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
		input_ids = inputs["input_ids"]
		labels = targets["input_ids"]
		outputs = model(input_ids=input_ids, labels=labels)

		# Or use retriever separately
		model = transformers.RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
		# 1. Encode
		question_hidden_states = model.question_encoder(input_ids)[0]
		# 2. Retrieve
		docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
		doc_scores = torch.bmm(
			question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
		).squeeze(1)
		# 3. Forward to generator
		outputs = model(
			context_input_ids=docs_dict["context_input_ids"],
			context_attention_mask=docs_dict["context_attention_mask"],
			doc_scores=doc_scores,
			decoder_input_ids=labels,
		)

		# Or directly generate
		generated = model.generate(
			context_input_ids=docs_dict["context_input_ids"],
			context_attention_mask=docs_dict["context_attention_mask"],
			doc_scores=doc_scores,
		)
		generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

# REF [site] >> https://github.com/microsoft/CodeBERT
def codebert_example():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if True:
		if False:
			tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base")
			model = transformers.RobertaModel.from_pretrained("microsoft/codebert-base")
		else:
			tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/codebert-base")
			model = transformers.AutoModel.from_pretrained("microsoft/codebert-base")
		model.to(device)

		nl_tokens = tokenizer.tokenize("return maximum value")
		# ['return', 'Ġmaximum', 'Ġvalue'].

		code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
		# ['def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb'].

		tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
		# ['<s>', 'return', 'Ġmaximum', 'Ġvalue', '</s>', 'def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb', '</s>'].

		tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
		# [0, 30921, 4532, 923, 2, 9232, 19220, 1640, 102, 6, 428, 3256, 114, 10, 15698, 428, 35, 671, 10, 1493, 671, 741, 2].

		context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]  # [1, 23, 768].

	if True:
		# CodeBERT is not suitable for mask prediction task, while CodeBERT (MLM) is suitable for mask prediction task.

		model = transformers.RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
		tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
		fill_mask = transformers.pipeline("fill-mask", model=model, tokenizer=tokenizer)

		CODE = "if (x is not None) <mask> (x>1)"
		outputs = fill_mask(CODE)

		print(outputs)
		"""
		Output: s'and', 'or', 'if', 'then', 'AND'.
		The detailed outputs:
		{'sequence': '<s> if (x is not None) and (x>1)</s>', 'score': 0.6049249172210693, 'token': 8}
		{'sequence': '<s> if (x is not None) or (x>1)</s>', 'score': 0.30680200457572937, 'token': 50}
		{'sequence': '<s> if (x is not None) if (x>1)</s>', 'score': 0.02133703976869583, 'token': 114}
		{'sequence': '<s> if (x is not None) then (x>1)</s>', 'score': 0.018607674166560173, 'token': 172}
		{'sequence': '<s> if (x is not None) AND (x>1)</s>', 'score': 0.007619690150022507, 'token': 4248}
		"""

# REF [site] >> https://huggingface.co/huggingface/CodeBERTa-language-id
def codeberta_example():
	# Models:
	#	huggingface/CodeBERTa-small-v1.
	#	huggingface/CodeBERTa-language-id.

	if True:
		# Masked language modeling prediction.

		fill_mask = transformers.pipeline(
			"fill-mask",
			model="huggingface/CodeBERTa-small-v1",
			tokenizer="huggingface/CodeBERTa-small-v1"
		)

		# PHP.
		PHP_CODE = """
public static <mask> set(string $key, $value) {
	if (!in_array($key, self::$allowedKeys)) {
		throw new \InvalidArgumentException('Invalid key given');
	}
	self::$storedValues[$key] = $value;
}
""".lstrip()

		outputs = fill_mask(PHP_CODE)
		"""
		Top 5 predictions:
		' function' # prob 0.9999827146530151
		'function'  # 
		' void'     # 
		' def'      # 
		' final'    #
		"""

		# Python.
		PYTHON_CODE = """
def pipeline(
	task: str,
	model: Optional = None,
	framework: Optional[<mask>] = None,
	**kwargs
) -> Pipeline:
	pass
""".lstrip()

		outputs = fill_mask(PYTHON_CODE)
		# Top 5 predictions: 'framework', 'Framework', ' framework', 'None', 'str'.

		# Natural language (not code).
		outputs = fill_mask("My name is <mask>.")
		# {'sequence': '<s> My name is undefined.</s>', 'score': 0.2548016905784607, 'token': 3353}
		# {'sequence': '<s> My name is required.</s>', 'score': 0.07290805131196976, 'token': 2371}
		# {'sequence': '<s> My name is null.</s>', 'score': 0.06323737651109695, 'token': 469}
		# {'sequence': '<s> My name is name.</s>', 'score': 0.021919190883636475, 'token': 652}
		# {'sequence': '<s> My name is disabled.</s>', 'score': 0.019681859761476517, 'token': 7434}

	if False:
		# Programming language identification.
		# Using the raw model.

		CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"

		tokenizer = transformers.RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
		model = transformers.RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)

		input_ids = tokenizer.encode(CODE_TO_IDENTIFY)

		logits = model(input_ids)[0]

		language_idx = logits.argmax()  # Index for the resulting label.

	if True:
		# Programming language identification.
		# Using Pipelines.

		CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"

		pipeline = transformers.TextClassificationPipeline(
			model=transformers.RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID),
			tokenizer=transformers.RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
		)

		#outputs = pipeline(CODE_TO_IDENTIFY)

		outputs = pipeline("""
def f(x):
	return x**2
""")
		# [{'label': 'python', 'score': 0.9999965}].

		outputs = pipeline("const foo = 'bar'")
		# [{'label': 'javascript', 'score': 0.9977546}].

		outputs = pipeline("foo = 'bar'")
		# [{'label': 'javascript', 'score': 0.7176245}].

		outputs = pipeline("foo = u'bar'")
		# [{'label': 'python', 'score': 0.7638422}]

		outputs = pipeline("echo $FOO")
		# [{'label': 'php', 'score': 0.9995257}]

		outputs = pipeline("outcome := rand.Intn(6) + 1")
		# [{'label': 'go', 'score': 0.9936151}]

		outputs = pipeline(":=")
		# [{'label': 'go', 'score': 0.9998052}]

# REF [site] >> https://huggingface.co/Salesforce
def codet5_example():
	# Models:
	#	Salesforce/codet5-small.
	#	Salesforce/codet5-base: ~892MB.
	#	Salesforce/codet5-base-multi-sum: ~892MB.
	#	Salesforce/codet5-base-codexglue-sum-python.
	#	Salesforce/codet5-base-codexglue-sum-go.
	#	Salesforce/codet5-base-codexglue-sum-php.
	#	Salesforce/codet5-base-codexglue-sum-javascript.
	#	Salesforce/codet5-base-codexglue-sum-java.
	#	Salesforce/codet5-base-codexglue-sum-ruby.
	#	Salesforce/codet5-base-codexglue-clone.
	#	Salesforce/codet5-base-codexglue-concode.
	#	Salesforce/codet5-base-codexglue-defect.
	#	Salesforce/codet5-base-codexglue-refine-small.
	#	Salesforce/codet5-base-codexglue-refine-medium.
	#	Salesforce/codet5-base-codexglue-translate-cs-java.
	#	Salesforce/codet5-base-codexglue-translate-java-cs.
	#	Salesforce/codet5-large.
	#	Salesforce/codet5-large-ntp-py: ~1.48GB.

	if True:
		tokenizer = transformers.RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
		model = transformers.T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

		text = "def greet(user): print(f'hello <extra_id_0>!')"
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		# Simply generate a single sequence.
		generated_ids = model.generate(input_ids, max_length=8)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))  # Outputs: "{user.username}".

	if True:
		tokenizer = transformers.RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
		model = transformers.T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-multi-sum")

		text = """def svg_to_image(string, size=None):
	if isinstance(string, unicode):
		string = string.encode("utf-8")
		renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
	if not renderer.isValid():
		raise ValueError("Invalid SVG data.")
	if size is None:
		size = renderer.defaultSize()
		image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
		painter = QtGui.QPainter(image)
		renderer.render(painter)
	return image"""

		input_ids = tokenizer(text, return_tensors="pt").input_ids

		generated_ids = model.generate(input_ids, max_length=20)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))  # Output: "Convert a SVG string to a QImage.".

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
		model = transformers.T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py")

		text = "def hello_world():"
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		# Simply generate a single sequence.
		generated_ids = model.generate(input_ids, max_length=128)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))  # Output: "print("Hello World!")".

# REF [site] >> https://huggingface.co/Salesforce
def codet5p_example():
	# Models:
	#	Salesforce/codet5p-220m.
	#	Salesforce/codet5p-220m-py.
	#	Salesforce/codet5p-770m.
	#	Salesforce/codet5p-770m-py.
	#	Salesforce/codet5p-2b.
	#	Salesforce/codet5p-6b.
	#	Salesforce/codet5p-16b.
	#	Salesforce/instructcodet5p-16b.

	if True:
		from transformers import T5ForConditionalGeneration, AutoTokenizer

		checkpoint = "Salesforce/codet5p-770m"
		device = "cuda"  # For GPU usage or "cpu" for CPU usage.

		tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

		inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)

		outputs = model.generate(inputs, max_length=10)

		print(tokenizer.decode(outputs[0], skip_special_tokens=True))
		# Output: print "Hello World".

	if True:
		checkpoint = "Salesforce/codet5p-770m-py"
		device = "cuda"  # For GPU usage or "cpu" for CPU usage.

		tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
		model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

		inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)

		outputs = model.generate(inputs, max_length=10)

		print(tokenizer.decode(outputs[0], skip_special_tokens=True))
		# Output: print('Hello World!').

	if True:
		checkpoint = "Salesforce/instructcodet5p-16b"
		device = "cuda"  # For GPU usage or "cpu" for CPU usage.

		tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
		model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
			checkpoint,
			torch_dtype=torch.float16,
			low_cpu_mem_usage=True,
			trust_remote_code=True
		).to(device)

		encoding = tokenizer("def print_hello_world():", return_tensors="pt").to(device)
		encoding['decoder_input_ids'] = encoding['input_ids'].clone()

		outputs = model.generate(**encoding, max_length=15)

		print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# REF [site] >> https://huggingface.co/Salesforce
def codegen_example():
	# Models:
	#	Salesforce/codegen-350M-nl.
	#	Salesforce/codegen-350M-multi.
	#	Salesforce/codegen-350M-mono.
	#	Salesforce/codegen-2B-nl.
	#	Salesforce/codegen-2B-multi.
	#	Salesforce/codegen-2B-mono: ~5.69GB.
	#	Salesforce/codegen-6B-nl.
	#	Salesforce/codegen-6B-multi.
	#	Salesforce/codegen-6B-mono.
	#	Salesforce/codegen-16B-nl.
	#	Salesforce/codegen-16B-multi.
	#	Salesforce/codegen-16B-mono.

	pretrained_model_name = "Salesforce/codegen-2B-mono"

	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)
	model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name)

	text = "def hello_world():"
	input_ids = tokenizer(text, return_tensors="pt").input_ids

	generated_ids = model.generate(input_ids, max_length=128)

	print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

# REF [site] >> https://huggingface.co/Salesforce
def codegen2_example():
	# Models:
	#	Salesforce/codegen2-1B.
	#	Salesforce/codegen2-3_7B.
	#	Salesforce/codegen2-7B.
	#	Salesforce/codegen2-16B.

	if True:
		# Causal sampling.

		tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
		model = transformers.AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

		text = "def hello_world():"
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		generated_ids = model.generate(input_ids, max_length=128)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

	if True:
		"""
		Infill sampling.

		For infill sampling, we introduce three new special token types:
			<mask_N>: N-th span to be masked. In practice, use <mask_1> to where you want to sample infill.
			<sep>: Separator token between the suffix and the infilled sample. See below.
			<eom>: "End-Of-Mask" token that model will output at the end of infilling. You may use this token to truncate the output.
		"""

		tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
		model = transformers.AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

		"""
		If we want to generate infill for the following cursor position of a function:

		def hello_world():
			|
			return name
		"""

		def format(prefix, suffix):
			return prefix + "<mask_1>" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"

		prefix = "def hello_world():\n    "
		suffix = "    return name"
		text = format(prefix, suffix)
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		generated_ids = model.generate(input_ids, max_length=128)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=False)[len(text):])

# REF [site] >> https://huggingface.co/Salesforce
def codegen25_example():
	# Models:
	#	Salesforce/codegen25-7b-multi.
	#	Salesforce/codegen25-7b-mono.
	#	Salesforce/codegen25-7b-instruct.

	# Install:
	#	pip install tiktoken==0.4.0

	if True:
		# Causal sampling (code autocompletion).

		tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi", trust_remote_code=True)
		model = transformers.AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-multi")

		text = "def hello_world():"
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		generated_ids = model.generate(input_ids, max_length=128)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

	if True:
		"""
		Infill sampling.

		For infill sampling, we follow the CodeGen2 format:
			<mask_N>: N-th span to be masked. In practice, use <mask_1> to where you want to sample infill.
			<sep>: Separator token between the suffix and the infilled sample. See below.
			<eom>: "End-Of-Mask" token that model will output at the end of infilling. You may use this token to truncate the output.
		"""

		tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi", trust_remote_code=True)
		model = transformers.AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-multi")

		"""
		If we want to generate infill for the following cursor position of a function:

		def hello_world():
			|
			return name
		"""

		def format(prefix, suffix):
			return prefix + "<mask_1>" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"

		prefix = "def hello_world():\n    "
		suffix = "    return name"
		text = format(prefix, suffix)
		input_ids = tokenizer(text, return_tensors="pt").input_ids

		generated_ids = model.generate(input_ids, max_length=128)

		print(tokenizer.decode(generated_ids[0], skip_special_tokens=False)[len(text):])

# REF [site] >> https://huggingface.co/codeparrot
def codeparrot_example():
	# Models:
	#	codeparrot/codeparrot: ~6.17GB.
	#	codeparrot/codeparrot-small.
	#	codeparrot/codeparrot-small-multi.
	#	codeparrot/codeparrot-small-text-to-code.
	#	codeparrot/codeparrot-small-code-to-text.
	#	codeparrot/codeparrot-small-complexity-prediction.

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("codeparrot/codeparrot")
		model = transformers.AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot")

		inputs = tokenizer("def hello_world():", return_tensors="pt")
		outputs = model(**inputs)

		logits = outputs.logits  # [batch size, 5(?), vocab size].

		predicted_label = logits.argmax(-1)  # ???
		print(f"Predicted label: {predicted_label}.")
		#predicted_token = tokenizer.convert_ids_to_tokens(predicted_label[0])
		#print(f"Predicted token: {predicted_token}.")

	if True:
		pipe = transformers.pipeline("text-generation", model="codeparrot/codeparrot")
		outputs = pipe("def hello_world():")

		print("Prediction:")
		#print(outputs)
		print(outputs[0]["generated_text"])

# REF [site] >> https://huggingface.co/codellama
def code_llama_example():
	# Models:
	#	codellama/CodeLlama-7b-hf: ~13.48GB.
	#	codellama/CodeLlama-13b-hf.
	#	codellama/CodeLlama-34b-hf.
	#	codellama/CodeLlama-7b-Python-hf.
	#	codellama/CodeLlama-13b-Python-hf.
	#	codellama/CodeLlama-34b-Python-hf.
	#	codellama/CodeLlama-7b-Instruct-hf.
	#	codellama/CodeLlama-13b-Instruct-hf.
	#	codellama/CodeLlama-34b-Instruct-hf.

	model = "codellama/CodeLlama-7b-hf"

	tokenizer = transformers.AutoTokenizer.from_pretrained(model)
	pipeline = transformers.pipeline(
		"text-generation",
		model=model,
		torch_dtype=torch.float16,
		device_map="auto",
	)

	sequences = pipeline(
		"import socket\n\ndef ping_exponential_backoff(host: str):",
		do_sample=True,
		top_k=10,
		temperature=0.1,
		top_p=0.95,
		num_return_sequences=1,
		eos_token_id=tokenizer.eos_token_id,
		max_length=200,
	)
	for seq in sequences:
		print(f"Result: {seq['generated_text']}")

# REF [site] >> https://huggingface.co/bigcode
def star_coder_example():
	# Models:
	#	bigcode/starcoder.
	#	bigcode/starcoderplus.

	checkpoint = "bigcode/starcoder"
	device = "cuda"  # for GPU usage or "cpu" for CPU usage

	tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
	model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

	# Generation
	inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
	outputs = model.generate(inputs)
	print(tokenizer.decode(outputs[0]))

	# Fill-in-the-middle
	# Fill-in-the-middle uses special tokens to identify the prefix/middle/suffix part of the input and output:
	input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
	inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
	outputs = model.generate(inputs)
	print(tokenizer.decode(outputs[0]))

# REF [site] >> https://huggingface.co/replit
def replit_example():
	# Models:
	#	replit/replit-code-v1-3b.
	#	replit/replit-code-v1_5-3b.

	model_name = "replit/replit-code-v1-3b"

	if True:
		device = "cuda:0"

		if False:
			# Load model
			model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
			model.to(device=device)
		else:
			config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
			config.attn_config["attn_impl"] = "triton"

			# Load model
			model = transformers.AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
			model.to(device=device, dtype=torch.bfloat16)

		# Forward pass
		x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
		x = x.to(device=device)
		y = model(x)

		# Load tokenizer
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

		# Single input encoding + generation
		x = tokenizer.encode('def hello():\n  print("hello world")\n', return_tensors="pt").to(device=device)
		y = model.generate(x)

		# Decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
		generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
		print(generated_code)

	if True:
		# Generation
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		if True:
			model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
		elif False:
			# Loading in 8-bit
			model = AutoModelForCausalLM.from_pretrained(
				model_name, 
				trust_remote_code=True, 
				device_map="auto",
				load_in_8bit=True
			)
		elif False:
			# Loading in 4-bit
			model = AutoModelForCausalLM.from_pretrained(
				model_name, 
				trust_remote_code=True, 
				device_map="auto",
				load_in_4bit=True
			)

		x = tokenizer.encode("def fibonacci(n): ", return_tensors="pt")
		y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

		# Decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
		generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
		print(generated_code)

# REF [site] >> https://huggingface.co/microsoft
def phi_example():
	# Models:
	#	microsoft/phi-1: ~2.84GB.
	#	microsoft/phi-1_5: ~2.84GB.

	torch.set_default_device("cuda")

	if False:
		model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/phi-1", trust_remote_code=True)
		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-1", trust_remote_code=True)

		inputs = tokenizer('''def print_prime(n):
	"""
	Print all primes between 1 and n
	"""''', return_tensors="pt", return_attention_mask=False)

		if True:
			outputs = model.generate(**inputs, max_length=200)
		else:
			with torch.autocast(model.device.type, dtype=torch.float16, enabled=True):
				outputs = model.generate(**inputs, max_length=200)

		text = tokenizer.batch_decode(outputs)[0]
		print(text)

	if True:
		model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

		inputs = tokenizer('''```python
def print_prime(n):
	"""
	Print all primes between 1 and n
	"""''', return_tensors="pt", return_attention_mask=False)

		if True:
			outputs = model.generate(**inputs, max_length=200)
		else:
			with torch.autocast(model.device.type, dtype=torch.float16, enabled=True):
				outputs = model.generate(**inputs, max_length=200)

		text = tokenizer.batch_decode(outputs)[0]
		print(text)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vit
def vit_example():
	import datasets

	if False:
		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
		model = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

		#-----
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
		model = transformers.ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

		num_patches = (model.config.image_size // model.config.patch_size)**2
		pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
		# Create random boolean mask of shape (batch_size, num_patches).
		bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

		outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

		loss, reconstructed_pixel_values = outputs.loss, outputs.logits
		print(reconstructed_pixel_values.shape)

	if True:
		# Masked image modeling.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
		model = transformers.ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

		num_patches = (model.config.image_size // model.config.patch_size)**2
		pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
		# Create random boolean mask of shape (batch_size, num_patches).
		bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

		outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

		loss, reconstructed_pixel_values = outputs.loss, outputs.logits
		print(reconstructed_pixel_values.shape)

	if True:
		# Image classification.

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
		model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			logits = model(**inputs).logits

		# Model predicts one of the 1000 ImageNet classes.
		predicted_label = logits.argmax(-1).item()
		print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/deit
def deit_example():
	import datasets

	if False:
		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
		model = transformers.DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Masked image modeling.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
		model = transformers.DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

		num_patches = (model.config.image_size // model.config.patch_size) ** 2
		pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
		# Create random boolean mask of shape (batch_size, num_patches).
		bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

		outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

		loss, reconstructed_pixel_values = outputs.loss, outputs.logits
		print(reconstructed_pixel_values.shape)

	if True:
		# Image classification.

		torch.manual_seed(3)

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		# Note: we are loading a DeiTForImageClassificationWithTeacher from the hub here,
		# so the head will be randomly initialized, hence the predictions will be random.
		image_processor = transformers.DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
		model = transformers.DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

		inputs = image_processor(images=image, return_tensors="pt")
		outputs = model(**inputs)

		logits = outputs.logits
		# Model predicts one of the 1000 ImageNet classes.
		predicted_class_idx = logits.argmax(-1).item()
		print("Predicted class: {}.".format(model.config.id2label[predicted_class_idx]))

	if True:
		# Image classification with teacher.

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
		model = transformers.DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			logits = model(**inputs).logits

		# Model predicts one of the 1000 ImageNet classes.
		predicted_label = logits.argmax(-1).item()
		print(model.config.id2label[predicted_label])

# REF [site] >>
#	https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
#	https://huggingface.co/facebook
def dino_example():
	# Models:
	#	facebook/dino-vits8.
	#	facebook/dino-vits16.
	#	facebook/dino-vitb8.
	#	facebook/dino-vitb16.
	#	facebook/dinov2-small.
	#	facebook/dinov2-base.
	#	facebook/dinov2-large.
	#	facebook/dinov2-giant.
	#	facebook/dinov2-small-imagenet1k-1-layer.
	#	facebook/dinov2-base-imagenet1k-1-layer.
	#	facebook/dinov2-large-imagenet1k-1-layer.
	#	facebook/dinov2-giant-imagenet1k-1-layer.

	if False:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
		model = transformers.ViTModel.from_pretrained("facebook/dino-vitb16")

		inputs = processor(images=image, return_tensors="pt")
		outputs = model(**inputs)
		last_hidden_states = outputs.last_hidden_state

	if False:
		# Initializing a Dinov2 dinov2-base-patch16-224 style configuration
		configuration = transformers.Dinov2Config()

		# Initializing a model (with random weights) from the dinov2-base-patch16-224 style configuration
		model = transformers.Dinov2Model(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		import datasets

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		image_processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-base")
		model = transformers.Dinov2Model.from_pretrained("facebook/dinov2-base")

		inputs = image_processor(image, return_tensors="pt")

		with torch.no_grad():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		list(last_hidden_states.shape)

	if True:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-base")
		model = transformers.AutoModel.from_pretrained("facebook/dinov2-base")

		inputs = processor(images=image, return_tensors="pt")
		outputs = model(**inputs)
		last_hidden_states = outputs[0]

		# We have to force return_dict=False for tracing
		model.config.return_dict = False

		with torch.no_grad():
			traced_model = torch.jit.trace(model, [inputs.pixel_values])
			traced_outputs = traced_model(inputs.pixel_values)

		print((last_hidden_states - traced_outputs[0]).abs().max())

	if True:
		import datasets

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		image_processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
		model = transformers.Dinov2ForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")

		inputs = image_processor(image, return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		# Model predicts one of the 1000 ImageNet classes
		predicted_label = logits.argmax(-1).item()
		print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/facebook
def dpt_example():
	# Models:
	#	facebook/dpt-dinov2-small-kitti.
	#	facebook/dpt-dinov2-small-nyu.
	#	facebook/dpt-dinov2-base-kitti.
	#	facebook/dpt-dinov2-base-nyu.
	#	facebook/dpt-dinov2-large-kitti.
	#	facebook/dpt-dinov2-large-nyu.
	#	facebook/dpt-dinov2-giant-kitti.
	#	facebook/dpt-dinov2-giant-nyu.

	import numpy as np

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = transformers.AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-base-nyu")
	model = transformers.DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-base-nyu")

	# Prepare image for the model
	inputs = image_processor(images=image, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs)
		predicted_depth = outputs.predicted_depth

	# Interpolate to original size
	prediction = torch.nn.functional.interpolate(
		predicted_depth.unsqueeze(1),
		size=image.size[::-1],
		mode="bicubic",
		align_corners=False,
	)

	# Visualize the prediction
	output = prediction.squeeze().cpu().numpy()
	formatted = (output * 255 / np.max(output)).astype("uint8")
	depth = Image.fromarray(formatted)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_example():
	if False:
		# Prepare image and text.
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		text = "hello world"

		processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
		model = transformers.ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

		inputs = processor(image, text, return_tensors="pt")
		outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Masked LM.

		import re

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		text = "a bunch of [MASK] laying on a [MASK]."

		processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
		model = transformers.ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

		# Prepare inputs.
		encoding = processor(image, text, return_tensors="pt")

		# Forward pass.
		outputs = model(**encoding)

		tl = len(re.findall("\[MASK\]", text))
		inferred_token = [text]

		# Gradually fill in the MASK tokens, one by one.
		model.eval()
		with torch.no_grad():
			for i in range(tl):
				encoded = processor.tokenizer(inferred_token)
				input_ids = torch.tensor(encoded.input_ids)
				encoded = encoded["input_ids"][0][1:-1]
				outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
				mlm_logits = outputs.logits[0]  # Shape (seq_len, vocab_size).
				# Only take into account text features (minus CLS and SEP token).
				mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
				mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
				# Only take into account text.
				mlm_values[torch.tensor(encoded) != 103] = 0
				select = mlm_values.argmax().item()
				encoded[select] = mlm_ids[select].item()
				inferred_token = [processor.decode(encoded)]

		selected_token = ""
		encoded = processor.tokenizer(inferred_token)
		output = processor.decode(encoded.input_ids[0], skip_special_tokens=True)
		print(output)

	if True:
		# Question answering.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		text = "How many cats are there?"

		processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
		model = transformers.ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

		# Prepare inputs.
		encoding = processor(image, text, return_tensors="pt")

		# Forward pass.
		outputs = model(**encoding)

		logits = outputs.logits
		idx = logits.argmax(-1).item()
		print("Predicted answer:", model.config.id2label[idx])

	if True:
		# Images and text classification.

		image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
		image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg", stream=True).raw)
		text = "The left image contains twice the number of dogs as the right image."

		processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
		model = transformers.ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

		# Prepare inputs.
		encoding = processor([image1, image2], text, return_tensors="pt")

		# Forward pass.
		outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))

		logits = outputs.logits
		idx = logits.argmax(-1).item()
		print("Predicted answer: {}.".format(model.config.id2label[idx]))

	if True:
		# Image and text retrieval.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

		processor = transformers.ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
		model = transformers.ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

		# Forward pass.
		scores = dict()
		for text in texts:
			# Prepare inputs.
			encoding = processor(image, text, return_tensors="pt")
			outputs = model(**encoding)
			scores[text] = outputs.logits[0, :].item()

		print(scores)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/beit
def beit_example():
	import datasets

	if False:
		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
		model = transformers.BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)

	if True:
		# Masked image modeling.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
		model = transformers.BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

		num_patches = (model.config.image_size // model.config.patch_size)**2
		pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
		# Create random boolean mask of shape (batch_size, num_patches).
		bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

		outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

		loss, logits = outputs.loss, outputs.logits
		print(logits.shape)

	if True:
		# Image classification.

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		feature_extractor = transformers.BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
		model = transformers.BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

		inputs = feature_extractor(image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			logits = model(**inputs).logits

		# Model predicts one of the 1000 ImageNet classes.
		predicted_label = logits.argmax(-1).item()
		print(model.config.id2label[predicted_label])

	if True:
		# Semantic segmentation.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
		model = transformers.BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

		inputs = image_processor(images=image, return_tensors="pt")
		outputs = model(**inputs)

		# Logits are of shape (batch_size, num_labels, height, width).
		logits = outputs.logits
		print(logits.shape)

# REF [site] >>
#	https://huggingface.co/openai
#	https://huggingface.co/docs/transformers/main/model_doc/clip
def clip_example():
	# Models:
	#	openai/clip-vit-base-patch16.
	#	openai/clip-vit-base-patch32: ~605MB.
	#	openai/clip-vit-large-patch14.
	#	openai/clip-vit-large-patch14-336.

	pretrained_model_name = "openai/clip-vit-base-patch32"

	if True:
		model = transformers.CLIPModel.from_pretrained(pretrained_model_name)
		processor = transformers.CLIPProcessor.from_pretrained(pretrained_model_name)

		# model.text_model
		# model.text_projection
		# model.vision_model
		# model.visual_projection

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		texts = ["a photo of a cat", "a photo of a dog"]
		inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
		print(f"Input: {inputs}.")  # {"input_ids", "attention_mask", "pixel_values"}.

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)  # {loss, logits_per_image, logits_per_text, text_embeds, image_embeds, text_model_output, vision_model_output}.

		# outputs.text_model_output.last_hidden_state
		# outputs.text_model_output.pooler_output
		# outputs.vision_model_output.last_hidden_state
		# outputs.vision_model_output.pooler_output

		logits_per_image = outputs.logits_per_image  # This is the image-text similarity score.
		probs = logits_per_image.softmax(dim=1)  # We can take the softmax to get the label probabilities.
		print(f"Probability: {probs}.")

	if True:
		# Text embedding.
		model = transformers.CLIPTextModelWithProjection.from_pretrained(pretrained_model_name)
		tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

		inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		text_embeds = outputs.text_embeds
		print(f"{text_embeds=}")

		# Image embedding.
		model = transformers.CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name)
		processor = transformers.AutoProcessor.from_pretrained(pretrained_model_name)

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = processor(images=image, return_tensors="pt")

		with torch.no_grad():
			outputs = model(**inputs)

		image_embeds = outputs.image_embeds
		print(f"{image_embeds=}")

	if True:
		model = transformers.CLIPModel.from_pretrained(pretrained_model_name)

		# Image embedding.
		processor = transformers.AutoProcessor.from_pretrained(pretrained_model_name)

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = processor(images=image, return_tensors="pt")

		image_features = model.get_image_features(**inputs)
		#image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
		print(f"{image_features=}")

		# Text embedding.
		tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

		inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

		text_features = model.get_text_features(**inputs)
		print(f"{text_features=}")

# REF [site] >>
#	https://huggingface.co/kakaobrain
#	https://huggingface.co/docs/transformers/main/model_doc/align
def align_example():
	# Models:
	#	kakaobrain/coyo-align-b7-base.
	#	kakaobrain/align-base.

	pretrained_model_name = "kakaobrain/align-base"

	if True:
		# Zero-shot image classification.

		processor = transformers.AlignProcessor.from_pretrained(pretrained_model_name)
		model = transformers.AlignModel.from_pretrained(pretrained_model_name)

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		texts = ["an image of a cat", "an image of a dog"]
		inputs = processor(text=texts, images=image, return_tensors="pt")
		print(f"Input: {inputs}.")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		logits_per_image = outputs.logits_per_image  # This is the image-text similarity score.
		probs = logits_per_image.softmax(dim=1)  # We can take the softmax to get the label probabilities.
		print(f"Probability: {probs}.")

	if True:
		# Multi-modal embedding retrieval.

		processor = transformers.AlignProcessor.from_pretrained(pretrained_model_name)
		model = transformers.AlignModel.from_pretrained(pretrained_model_name)

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		text = "an image of a cat"
		inputs = processor(text=text, images=image, return_tensors="pt")

		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)

		# Multi-modal text embedding.
		text_embeds = outputs.text_embeds  # torch.Tensor.
		print(f"{text_embeds=}")

		# Multi-modal image embedding.
		image_embeds = outputs.image_embeds  # torch.Tensor.
		print(f"{image_embeds=}")

	if True:
		# Multi-modal embedding retrieval.

		processor = transformers.AlignProcessor.from_pretrained(pretrained_model_name)
		model = transformers.AlignModel.from_pretrained(pretrained_model_name)

		# Image embeddings.
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = processor(images=image, return_tensors="pt")

		image_embeds = model.get_image_features(
			pixel_values=inputs["pixel_values"],
		)
		print(f"{image_embeds=}")

		# Text embeddings.
		text = "an image of a cat"
		inputs = processor(text=text, return_tensors="pt")

		text_embeds = model.get_text_features(
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
			token_type_ids=inputs["token_type_ids"],
		)
		print(f"{text_embeds=}")

# REF [site] >>
#	https://huggingface.co/microsoft
#	https://huggingface.co/docs/transformers/main/en/model_doc/git
def git_example():
	# Models:
	#	microsoft/git-base.
	#	microsoft/git-base-coco: ~707MB.
	#	microsoft/git-base-vqav2.
	#	microsoft/git-base-textvqa: ~709MB.
	#	microsoft/git-base-textcaps.
	#	microsoft/git-base-vatex: ~707MB.
	#	microsoft/git-base-msrvtt-qa.
	#	microsoft/git-large.
	#	microsoft/git-large-coco.
	#	microsoft/git-large-vqav2.
	#	microsoft/git-large-textvqa.
	#	microsoft/git-large-textcaps.
	#	microsoft/git-large-vatex.
	#	microsoft/git-large-msrvtt-qa.
	#	microsoft/git-large-r.
	#	microsoft/git-large-r-coco.
	#	microsoft/git-large-r-textcaps.

	if False:
		# Initializing a GitVisionConfig with microsoft/git-base style configuration.
		configuration = transformers.GitVisionConfig()

		# Initializing a GitVisionModel (with random weights) from the microsoft/git-base style configuration.
		model = transformers.GitVisionModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if False:
		processor = transformers.AutoProcessor.from_pretrained("microsoft/git-base")
		model = transformers.GitVisionModel.from_pretrained("microsoft/git-base")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		inputs = processor(images=image, return_tensors="pt")

		outputs = model(**inputs)
		last_hidden_state = outputs.last_hidden_state

	if False:
		# Initializing a GIT microsoft/git-base style configuration.
		configuration = transformers.GitConfig()

		# Initializing a model (with random weights) from the microsoft/git-base style configuration.

		# Accessing the model configuration.
		configuration = model.config

	if False:
		processor = transformers.AutoProcessor.from_pretrained("microsoft/git-base")
		model = transformers.AutoModel.from_pretrained("microsoft/git-base")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		text = "this is an image of two cats"

		inputs = processor(text, images=image, return_tensors="pt")

		outputs = model(**inputs)
		last_hidden_state = outputs.last_hidden_state

	if True:
		# Image captioning example.

		processor = transformers.AutoProcessor.from_pretrained("microsoft/git-base-coco")
		model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		pixel_values = processor(images=image, return_tensors="pt").pixel_values

		generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

		generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		print(generated_caption)  # Output: "two cats sleeping on a pink blanket next to remotes.".

	if True:
		# Visual question answering (VQA) example.

		import huggingface_hub

		processor = transformers.AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
		model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

		file_path = huggingface_hub.hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
		image = Image.open(file_path).convert("RGB")
		pixel_values = processor(images=image, return_tensors="pt").pixel_values

		question = "what does the front of the bus say at the top?"
		input_ids = processor(text=question, add_special_tokens=False).input_ids
		input_ids = [processor.tokenizer.cls_token_id] + input_ids
		input_ids = torch.tensor(input_ids).unsqueeze(0)

		generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)

		print(processor.batch_decode(generated_ids, skip_special_tokens=True))  # Output: "['what does the front of the bus say at the top? special']".

	if True:
		# Video captioning example.

		import numpy as np
		import huggingface_hub
		import av

		processor = transformers.AutoProcessor.from_pretrained("microsoft/git-base-vatex")
		model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

		# Set seed for reproducability.
		np.random.seed(45)

		def read_video_pyav(container, indices):
			'''
			Decode the video with PyAV decoder.
			Args:
				container (`av.container.input.InputContainer`): PyAV container.
				indices (`List[int]`): List of frame indices to decode.
			Returns:
				result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
			'''
			frames = []
			container.seek(0)
			start_index = indices[0]
			end_index = indices[-1]
			for i, frame in enumerate(container.decode(video=0)):
				if i > end_index:
					break
				if i >= start_index and i in indices:
					frames.append(frame)
			return np.stack([x.to_ndarray(format="rgb24") for x in frames])

		def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
			converted_len = int(clip_len * frame_sample_rate)
			end_idx = np.random.randint(converted_len, seg_len)
			start_idx = end_idx - converted_len
			indices = np.linspace(start_idx, end_idx, num=clip_len)
			indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
			return indices

		# Load video.
		file_path = huggingface_hub.hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
		container = av.open(file_path)

		# Sample frames.
		num_frames = model.config.num_image_with_embedding
		indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
		frames = read_video_pyav(container, indices)
		pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values

		generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

		print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))  # Output: "Generated caption: ['a woman is sitting at a table and she is talking about the food she is holding.']".

# REF [site] >> https://huggingface.co/Salesforce
def blip_example():
	# Models:
	#	Salesforce/blip-itm-base-coco.
	#	Salesforce/blip-itm-base-flickr: ~895MB.
	#	Salesforce/blip-itm-large-coco.
	#	Salesforce/blip-itm-large-flickr.
	#
	#	Salesforce/blip-vqa-base: ~1.54GB.
	#	Salesforce/blip-vqa-capfilt-large.
	#
	#	Salesforce/blip-image-captioning-base: ~990MB.
	#	Salesforce/blip-image-captioning-large.
	#
	#	Salesforce/blip2-flan-t5-xl: ~15.77GB.
	#	Salesforce/blip2-flan-t5-xl-coco.
	#	Salesforce/blip2-flan-t5-xxl.
	#	Salesforce/blip2-opt-2.7b.
	#	Salesforce/blip2-opt-2.7b-coco.
	#	Salesforce/blip2-opt-6.7b.
	#	Salesforce/blip2-opt-6.7b-coco.

	if True:
		processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip-itm-base-flickr")
		model = transformers.BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-flickr")
		#model = transformers.BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-flickr").to("cuda")
		#model = transformers.BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-flickr", torch_dtype=torch.float16).to("cuda")

		img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
		raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

		question = "A woman and a dog sitting together in a beach."
		inputs = processor(raw_image, question, return_tensors="pt")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

		itm_scores = model(**inputs)[0]  # {'itm_score', 'last_hidden_state', 'question_embeds'}.
		cosine_score = model(**inputs, use_itm_head=False)[0]

	if True:
		# Visual question answering (VQA).

		processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
		model = transformers.BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
		#model = transformers.BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
		#model = transformers.BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype=torch.float16).to("cuda")

		img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
		raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

		question = "how many dogs are in the picture?"
		inputs = processor(raw_image, question, return_tensors="pt")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

		out = model.generate(**inputs)

		print(processor.decode(out[0], skip_special_tokens=True))  # Output: "1".

	if True:
		# Image captioning.

		processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
		model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
		#model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
		#model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

		img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
		raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

		# Conditional image captioning.
		text = "a photography of"
		inputs = processor(raw_image, text, return_tensors="pt")
		#inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
		#inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

		out = model.generate(**inputs)

		print(processor.decode(out[0], skip_special_tokens=True))  # Output: "a photography of a woman and her dog".

		# Unconditional image captioning.
		inputs = processor(raw_image, return_tensors="pt")
		#inputs = processor(raw_image, return_tensors="pt").to("cuda")
		#inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

		out = model.generate(**inputs)

		print(processor.decode(out[0], skip_special_tokens=True))  # Output: "a woman sitting on the beach with her dog".

	if True:
		processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
		#processor = transformers.Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")  # Error.
		model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
		#model = transformers.Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")  # Error.
		#model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")
		#model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto")
		#model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")

		img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
		raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

		question = "how many dogs are in the picture?"
		inputs = processor(raw_image, question, return_tensors="pt")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
		#inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

		out = model.generate(**inputs)

		print(processor.decode(out[0], skip_special_tokens=True))

# REF [site] >> https://huggingface.co/openflamingo
def openflamingo_example():
	# Models:
	#	openflamingo/OpenFlamingo-3B-vitl-mpt1b.
	#	openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct.
	#	openflamingo/OpenFlamingo-4B-vitl-rpj3b.
	#	openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct.
	#	openflamingo/OpenFlamingo-9B-vitl-mpt7b.

	# Install:
	#	pip install open-flamingo

	from open_flamingo import create_model_and_transforms
	import huggingface_hub

	# Initialization.
	model, image_processor, tokenizer = create_model_and_transforms(
		clip_vision_encoder_path="ViT-L-14",
		clip_vision_encoder_pretrained="openai",
		lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
		tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
		cross_attn_every_n_layers=1
	)

	# Grab model checkpoint from huggingface hub.
	checkpoint_path = huggingface_hub.hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
	model.load_state_dict(torch.load(checkpoint_path), strict=False)

	# Generation example.

	# Step 1: Load images.
	demo_image_one = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
	demo_image_two = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw)
	query_image = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw)

	# Step 2: Preprocessing images.
	#	Details: For OpenFlamingo, we expect the image to be a torch tensor of shape batch_size x num_media x num_frames x channels x height x width. 
	#	In this case batch_size = 1, num_media = 3, num_frames = 1, channels = 3, height = 224, width = 224.
	vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
	vision_x = torch.cat(vision_x, dim=0)
	vision_x = vision_x.unsqueeze(1).unsqueeze(0)

	# Step 3: Preprocessing text.
	#	Details: In the text we expect an <image> special token to indicate where an image is.
	#	We also expect an <|endofchunk|> special token to indicate the end of the text portion associated with an image.
	tokenizer.padding_side = "left"  # For generation padding tokens should be on the left.
	lang_x = tokenizer(
		["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
		return_tensors="pt",
	)

	# Step 4: Generate text.
	generated_text = model.generate(
		vision_x=vision_x,
		lang_x=lang_x["input_ids"],
		attention_mask=lang_x["attention_mask"],
		max_new_tokens=20,
		num_beams=3,
	)

	print(f"Generated text: {tokenizer.decode(generated_text[0])}.")

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/tvlt
def tvlt_example():
	import numpy as np

	if False:
		# Initializing a TVLT ZinengTang/tvlt-base style configuration.
		configuration = transformers.TvltConfig()

		# Initializing a model (with random weights) from the ZinengTang/tvlt-base style configuration.
		model = transformers.TvltModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if True:
		processor = transformers.TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
		model = transformers.TvltModel.from_pretrained("ZinengTang/tvlt-base")

		num_frames = 8
		images = list(np.random.randn(num_frames, 3, 224, 224))
		audio = list(np.random.randn(10000))
		input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

		outputs = model(**input_dict)

		loss = outputs.loss

	if True:
		processor = transformers.TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
		model = transformers.TvltForPreTraining.from_pretrained("ZinengTang/tvlt-base")

		num_frames = 8
		images = list(np.random.randn(num_frames, 3, 224, 224))
		images_mixed = list(np.random.randn(num_frames, 3, 224, 224))
		audio = list(np.random.randn(10000))
		input_dict = processor(images, audio, images_mixed, sampling_rate=44100, mask_pixel=True, mask_audio=True, return_tensors="pt")

		outputs = model(**input_dict)

		loss = outputs.loss

	if True:
		processor = transformers.TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
		model = transformers.TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")

		num_frames = 8
		images = list(np.random.randn(num_frames, 3, 224, 224))
		audio = list(np.random.randn(10000))
		input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

		outputs = model(**input_dict)

		loss = outputs.loss

# REF [site] >> https://huggingface.co/microsoft/kosmos-2-patch14-224
def kosmos_example():
	if True:
		model = transformers.AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")  # ~6.66GB.
		processor = transformers.AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

		prompt = "<grounding>An image of"

		url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
		image = Image.open(requests.get(url, stream=True).raw)

		# The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
		image.save("./new_image.jpg")
		image = Image.open("./new_image.jpg")

		inputs = processor(text=prompt, images=image, return_tensors="pt")

		generated_ids = model.generate(
			pixel_values=inputs["pixel_values"],
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
			image_embeds=None,
			image_embeds_position_mask=inputs["image_embeds_position_mask"],
			use_cache=True,
			max_new_tokens=128,
		)
		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

		# Specify `cleanup_and_extract=False` in order to see the raw model generation.
		processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

		print(processed_text)
		# `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

		# By default, the generated text is cleanup and the entities are extracted.
		processed_text, entities = processor.post_process_generation(generated_text)

		print(processed_text)
		# `An image of a snowman warming himself by a fire.`

		print(entities)  # [(entity_name, (start, end), (x1_normalized, y1_normalized, x2_normalized, y2_normalized))]. (start, end): the start and end index of the noun chunk in caption.
		# `[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]`

	if False:
		model = transformers.AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")  # ~6.66GB.
		processor = transformers.AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

		url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
		image = Image.open(requests.get(url, stream=True).raw)

		def run_example(prompt):
			inputs = processor(text=prompt, images=image, return_tensors="pt")
			generated_ids = model.generate(
				pixel_values=inputs["pixel_values"],
				input_ids=inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				image_embeds=None,
				image_embeds_position_mask=inputs["image_embeds_position_mask"],
				use_cache=True,
				max_new_tokens=128,
			)
			generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
			_processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
			processed_text, entities = processor.post_process_generation(generated_text)

			print(processed_text)
			print(entities)
			print(_processed_text)

		#-----
		# Multimodal grounding.

		# Phrase grounding.
		prompt = "<grounding><phrase> a snowman</phrase>"
		run_example(prompt)

		# a snowman is warming himself by the fire
		# [('a snowman', (0, 9), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('the fire', (32, 40), [(0.203125, 0.015625, 0.453125, 0.859375)])]
		# <grounding><phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> is warming himself by<phrase> the fire</phrase><object><patch_index_0006><patch_index_0878></object>

		# Referring expression comprehension.
		prompt = "<grounding><phrase> a snowman next to a fire</phrase>"
		run_example(prompt)

		# a snowman next to a fire
		# [('a snowman next to a fire', (0, 24), [(0.390625, 0.046875, 0.984375, 0.828125)])]
		# <grounding><phrase> a snowman next to a fire</phrase><object><patch_index_0044><patch_index_0863></object>

		#-----
		# Multimodal referring.

		# Referring expression generation.
		prompt = "<grounding><phrase> It</phrase><object><patch_index_0044><patch_index_0863></object> is"
		run_example(prompt)

		# It is snowman in a hat and scarf
		# [('It', (0, 2), [(0.390625, 0.046875, 0.984375, 0.828125)])]
		# <grounding><phrase> It</phrase><object><patch_index_0044><patch_index_0863></object> is snowman in a hat and scarf

		#-----
		# Perception-language tasks.

		# Grounded VQA.
		prompt = "<grounding> Question: What is special about this image? Answer:"
		run_example(prompt)

		# Question: What is special about this image? Answer: The image features a snowman sitting by a campfire in the snow.
		# [('a snowman', (71, 80), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a campfire', (92, 102), [(0.109375, 0.640625, 0.546875, 0.984375)])]
		# <grounding> Question: What is special about this image? Answer: The image features<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> sitting by<phrase> a campfire</phrase><object><patch_index_0643><patch_index_1009></object> in the snow.

		# Grounded VQA with multimodal referring via bounding boxes.
		prompt = "<grounding> Question: Where is<phrase> the fire</phrase><object><patch_index_0005><patch_index_0911></object> next to? Answer:"
		run_example(prompt)

		# Question: Where is the fire next to? Answer: Near the snowman.
		# [('the fire', (19, 27), [(0.171875, 0.015625, 0.484375, 0.890625)]), ('the snowman', (50, 61), [(0.390625, 0.046875, 0.984375, 0.828125)])]
		# <grounding> Question: Where is<phrase> the fire</phrase><object><patch_index_0005><patch_index_0911></object> next to? Answer: Near<phrase> the snowman</phrase><object><patch_index_0044><patch_index_0863></object>.

		#-----
		# Grounded image captioning.

		# Brief.
		prompt = "<grounding> An image of"
		run_example(prompt)

		# An image of a snowman warming himself by a campfire.
		# [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a campfire', (41, 51), [(0.109375, 0.640625, 0.546875, 0.984375)])]
		# <grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a campfire</phrase><object><patch_index_0643><patch_index_1009></object>.

		# Detailed.
		prompt = "<grounding> Describe this image in detail:"
		run_example(prompt)

		# Describe this image in detail: The image features a snowman sitting by a campfire in the snow. He is wearing a hat, scarf, and gloves, with a pot nearby and a cup nearby. The snowman appears to be enjoying the warmth of the fire, and it appears to have a warm and cozy atmosphere.
		# [('a campfire', (71, 81), [(0.171875, 0.015625, 0.484375, 0.984375)]), ('a hat', (109, 114), [(0.515625, 0.046875, 0.828125, 0.234375)]), ('scarf', (116, 121), [(0.515625, 0.234375, 0.890625, 0.578125)]), ('gloves', (127, 133), [(0.515625, 0.390625, 0.640625, 0.515625)]), ('a pot', (140, 145), [(0.078125, 0.609375, 0.265625, 0.859375)]), ('a cup', (157, 162), [(0.890625, 0.765625, 0.984375, 0.984375)])]
		# <grounding> Describe this image in detail: The image features a snowman sitting by<phrase> a campfire</phrase><object><patch_index_0005><patch_index_1007></object> in the snow. He is wearing<phrase> a hat</phrase><object><patch_index_0048><patch_index_0250></object>,<phrase> scarf</phrase><object><patch_index_0240><patch_index_0604></object>, and<phrase> gloves</phrase><object><patch_index_0400><patch_index_0532></object>, with<phrase> a pot</phrase><object><patch_index_0610><patch_index_0872></object> nearby and<phrase> a cup</phrase><object><patch_index_0796><patch_index_1023></object> nearby. The snowman appears to be enjoying the warmth of the fire, and it appears to have a warm and cozy atmosphere.

	if False:
		import os
		import numpy as np
		import cv2
		import torchvision.transforms as T

		def is_overlapping(rect1, rect2):
			x1, y1, x2, y2 = rect1
			x3, y3, x4, y4 = rect2
			return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

		def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
			"""_summary_
			Args:
				image (_type_): image or image path
				collect_entity_location (_type_): _description_
			"""
			if isinstance(image, Image.Image):
				image_h = image.height
				image_w = image.width
				image = np.array(image)[:, :, [2, 1, 0]]
			elif isinstance(image, str):
				if os.path.exists(image):
					pil_img = Image.open(image).convert("RGB")
					image = np.array(pil_img)[:, :, [2, 1, 0]]
					image_h = pil_img.height
					image_w = pil_img.width
				else:
					raise ValueError(f"Invaild image path, {image}")
			elif isinstance(image, torch.Tensor):
				image_tensor = image.cpu()
				reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
				reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
				image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
				pil_img = T.ToPILImage()(image_tensor)
				image_h = pil_img.height
				image_w = pil_img.width
				image = np.array(pil_img)[:, :, [2, 1, 0]]
			else:
				raise ValueError(f"Invaild image format, {type(image)} for {image}")

			if len(entities) == 0:
				return image

			new_image = image.copy()
			previous_bboxes = []
			# Size of text
			text_size = 1
			# Thickness of text
			text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
			box_line = 3
			(c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
			base_height = int(text_height * 0.675)
			text_offset_original = text_height - base_height
			text_spaces = 3

			for entity_name, (start, end), bboxes in entities:
				for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
					orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
					# Draw bbox
					# Random color
					color = tuple(np.random.randint(0, 255, size=3).tolist())
					new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

					l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

					x1 = orig_x1 - l_o
					y1 = orig_y1 - l_o

					if y1 < text_height + text_offset_original + 2 * text_spaces:
						y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
						x1 = orig_x1 + r_o

					# Add text background
					(text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
					text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

					for prev_bbox in previous_bboxes:
						while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
							text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
							text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
							y1 += (text_height + text_offset_original + 2 * text_spaces)

							if text_bg_y2 >= image_h:
								text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
								text_bg_y2 = image_h
								y1 = image_h
								break

					alpha = 0.5
					for i in range(text_bg_y1, text_bg_y2):
						for j in range(text_bg_x1, text_bg_x2):
							if i < image_h and j < image_w:
								if j < text_bg_x1 + 1.35 * c_width:
									# Original color
									bg_color = color
								else:
									# White
									bg_color = [255, 255, 255]
								new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

					cv2.putText(new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA)
					# Previous_locations.append((x1, y1))
					previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

			pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
			if save_path:
				pil_image.save(save_path)
			if show:
				pil_image.show()

			return new_image

		# The same image from the previous code example
		url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
		image = Image.open(requests.get(url, stream=True).raw)

		# From the previous code example
		entities = [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]

		# Draw the bounding bboxes
		draw_entity_boxes_on_image(image, entities, show=True)

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

# REF [site] >> https://huggingface.co/microsoft
def trocr_example():
	# Models:
	#	microsoft/trocr-base-handwritten.
	#	microsoft/trocr-base-printed.
	#	microsoft/trocr-base-stage1: ~1.54GB.
	#	microsoft/trocr-small-handwritten.
	#	microsoft/trocr-small-printed.
	#	microsoft/trocr-small-stage1.
	#	microsoft/trocr-large-handwritten.
	#	microsoft/trocr-large-printed.
	#	microsoft/trocr-large-stage1.
	#	microsoft/trocr-base-str: ~1.34GB.
	#	microsoft/trocr-large-str.

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

# REF [site] >> https://huggingface.co/microsoft
def microsoft_asr_example():
	import datasets

	dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
	dataset = dataset.sort("id")
	sampling_rate = dataset.features["audio"].sampling_rate
	example_speech = dataset[0]["audio"]["array"]

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
	model = transformers.SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

	inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

	predicted_ids = model.generate(**inputs, max_length=100)

	transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
	print(transcription[0])

# REF [site] >> https://huggingface.co/nvidia
def nvidia_asr_example():
	# Models:
	#	en, es, uk, fr, it, de, pl, hr, be, ca, ua, ru, zh.
	#
	#	nvidia/stt_en_conformer_ctc_small.
	#	nvidia/stt_en_conformer_ctc_large.
	#	nvidia/stt_en_conformer_transducer_large.
	#	nvidia/stt_en_conformer_transducer_xlarge.
	#	nvidia/stt_en_fastconformer_ctc_large.
	#	nvidia/stt_en_fastconformer_ctc_xlarge.
	#	nvidia/stt_en_fastconformer_transducer_large.
	#	nvidia/stt_en_fastconformer_transducer_xlarge.
	#	nvidia/stt_en_fastconformer_transducer_xxlarge.
	#	nvidia/stt_en_fastconformer_hybrid_large_pc.
	#	nvidia/stt_en_citrinet_256_ls.
	#	nvidia/stt_en_citrinet_384_ls.
	#	nvidia/stt_en_citrinet_512_ls.
	#	nvidia/stt_en_citrinet_768_ls.
	#	nvidia/stt_en_citrinet_1024_ls.
	#	nvidia/stt_en_citrinet_1024_gamma_0_25.
	#
	#	nvidia/parakeet-rnnt-0.6b.
	#	nvidia/parakeet-rnnt-1.1b.
	#	nvidia/parakeet-ctc-0.6b.
	#	nvidia/parakeet-ctc-1.1b.
	#	nvidia/parakeet-tdt-1.1b.
	#
	#	nvidia/canary-1b.

	if False:
		# Install:
		#	pip install nemo_toolkit['all']

		#model_name = "nvidia/stt_en_fastconformer_transducer_large"
		model_name = "nvidia/parakeet-rnnt-1.1b"
		#model_name = "nvidia/parakeet-tdt-1.1b"

		import nemo.collections.asr as nemo_asr
		asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)

		# Get a sample file.
		#	wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
		transcribed = asr_model.transcribe(["./2086-149220-0033.wav"])
		print(transcribed)

	if False:
		# Install:
		#	pip install nemo_toolkit['all']

		model_name = "nvidia/parakeet-ctc-1.1b"

		import nemo.collections.asr as nemo_asr
		asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)

		# Get a sample file.
		#	wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
		transcribed = asr_model.transcribe(["./2086-149220-0033.wav"])
		print(transcribed)

	if True:
		# Install:
		#	pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]

		from nemo.collections.asr.models import EncDecMultiTaskModel

		# Load model
		canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")

		# Update dcode params
		decode_cfg = canary_model.cfg.decoding
		decode_cfg.beam.beam_size = 1
		canary_model.change_decoding_strategy(decode_cfg)

		if True:
			predicted_text = canary_model.transcribe(
				paths2audio_files=["path1.wav", "path2.wav"],
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

		if True:
			# Automatic speech-to-text recognition (ASR)
			'''
			# Example of a line in input_manifest.json
			{
				"audio_filepath": "/path/to/audio.wav",  # Path to the audio file
				"duration": 1000,  # Duration of the audio, can be set to `None` if using NeMo main branch
				"taskname": "asr",  # Use "s2t_translation" for speech-to-text translation with r1.23, or "ast" if using the NeMo main branch
				"source_lang": "en",  # Language of the audio input, set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
				"target_lang": "en",  # Language of the text output, choices=['en','de','es','fr']
				"pnc": "yes",  # Whether to have PnC output, choices=['yes', 'no']
				"answer": "na", 
			}
			'''

			predicted_text = canary_model.transcribe(
				"/path/to/input_manifest.json",
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

		if True:
			# Automatic speech-to-text translation (AST)
			'''
			# Example of a line in input_manifest.json
			{
				"audio_filepath": "/path/to/audio.wav",  # Path to the audio file
				"duration": 1000,  # Duration of the audio, can be set to `None` if using NeMo main branch
				"taskname": "s2t_translation", # r1.23 only recognizes "s2t_translation", but "ast" is supported if using the NeMo main branch
				"source_lang": "en", # Language of the audio input, choices=['en','de','es','fr']
				"target_lang": "de", # Language of the text output, choices=['en','de','es','fr']
				"pnc": "yes",  # Whether to have PnC output, choices=['yes', 'no']
				"answer": "na" 
			}
			'''

			predicted_text = canary_model.transcribe(
				"/path/to/input_manifest.json",
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

def openai_asr_example():
	# Models:
	#	openai/whisper-tiny.
	#	openai/whisper-base.
	#	openai/whisper-small.
	#	openai/whisper-medium.
	#	openai/whisper-large: ~6.17GB.
	#	openai/whisper-large-v2.

	import datasets

	pretrained_model_name = "openai/whisper-large"

	if True:
		# Transcription (English to English).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		model.config.forced_decoder_ids = None

		# Load dummy dataset and read audio files.
		ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
		sample = ds[0]["audio"]
		input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

		# Generate token ids.
		predicted_ids = model.generate(input_features)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
		print(f"Transcription (w/ special tokens): {transcription}.")
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription (w/o special tokens): {transcription}.")

	if False:
		# Transcription (French to French).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

		# Load streaming dataset and read first audio sample.
		ds = datasets.load_dataset("common_voice", "fr", split="test", streaming=True)
		ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
		input_speech = next(iter(ds))["audio"]
		input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

		# Generate token ids.
		predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids)
		print(f"Transcription (w/ special tokens): {transcription}.")
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription (w/o special tokens): {transcription}.")

	if True:
		# Translation (French to English).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

		# Load streaming dataset and read first audio sample.
		ds = datasets.load_dataset("common_voice", "fr", split="test", streaming=True)
		ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
		input_speech = next(iter(ds))["audio"]
		input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

		# Generate token ids.
		predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription: {transcription}.")

	if False:
		# Evaluation.

		import evaluate

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Device: {device}.")

		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)

		def map_to_pred(batch):
			audio = batch["audio"]
			input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
			batch["reference"] = processor.tokenizer._normalize(batch["text"])

			model.eval()
			with torch.no_grad():
				predicted_ids = model.generate(input_features.to(device))[0]
			transcription = processor.decode(predicted_ids)
			batch["prediction"] = processor.tokenizer._normalize(transcription)
			return batch

		librispeech_test_clean = datasets.load_dataset("librispeech_asr", "clean", split="test")
		result = librispeech_test_clean.map(map_to_pred)

		wer = evaluate.load("wer")
		print(f'WER = {100 * wer.compute(references=result["reference"], predictions=result["prediction"])}.')

	if False:
		# Long-form transcription.
		#	The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration.
		#	However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length.
		#	This is possible through Transformers pipeline method.
		#	Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline.
		#	It can also be extended to predict utterance level timestamps by passing return_timestamps=True.

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(f"Device: {device}.")

		pipe = transformers.pipeline(
			"automatic-speech-recognition",
			model=pretrained_model_name,
			chunk_length_s=30,
			device=device,
		)

		ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
		sample = ds[0]["audio"]

		prediction = pipe(sample.copy())["text"]
		print(f"Prediction: {prediction}.")

		# We can also return timestamps for the predictions.
		prediction = pipe(sample, return_timestamps=True)["chunks"]
		print(f"Prediction: {prediction}.")

# REF [site] >> https://huggingface.co/speechbrain
def speech_brain_asr_example():
	import speechbrain

	asr_model = speechbrain.pretrained.WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fr", savedir="pretrained_models/asr-whisper-large-v2-commonvoice-fr")
	#asr_model = speechbrain.pretrained.WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fr", savedir="pretrained_models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device": "cuda"})

	# https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fr/tree/main
	asr_model.transcribe_file("./example-fr.mp3")

# REF [site] >> https://huggingface.co/microsoft
def microsoft_tts_example():
	import datasets
	import soundfile as sf

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
	model = transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
	vocoder = transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

	inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

	# Load xvector containing speaker's voice characteristics from a dataset.
	embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
	speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

	speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

	sf.write("./speecht5_tts.wav", speech.numpy(), samplerate=16000)

# REF [site] >> https://huggingface.co/nvidia
def nvidia_tts_example():
	# Models:
	#	nvidia/tts_hifigan.
	#	nvidia/tts_en_fastpitch.

	# Install:
	#	pip install nemo_toolkit['all']

	import soundfile as sf

	# Load FastPitch.
	from nemo.collections.tts.models import FastPitchModel
	spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")

	# Load vocoder.
	from nemo.collections.tts.models import HifiGanModel
	model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

	# Generate audio.
	parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
	spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
	audio = model.convert_spectrogram_to_audio(spec=spectrogram)

	# Save the audio to disk in a file called speech.wav
	sf.write("./speech.wav", audio.to("cpu").numpy(), 22050)

# REF [site] >> https://huggingface.co/speechbrain
def speech_brain_tts_example():
	import speechbrain
	import torchaudio

	if True:
		# Intialize TTS (tacotron2) and Vocoder (HiFiGAN).
		tacotron2 = speechbrain.pretrained.Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
		hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
		#tacotron2 = speechbrain.pretrained.Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device": "cuda"})
		#hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device": "cuda"})

		# Running the TTS.
		if True:
			input_text = "Mary had a little lamb"
		else:
			input_text = [
				"A quick brown fox jumped over the lazy dog",
				"How much wood would a woodchuck chuck?",
				"Never odd or even"
			]
		mel_output, mel_length, alignment = tacotron2.encode_batch(input_text)  # torch.Tensor.

		# Running Vocoder (spectrogram-to-waveform).
		waveforms = hifi_gan.decode_batch(mel_output)  # torch.Tensor.

		# Save the waverform.
		torchaudio.save('./speechbrain_tts_tacotron2.wav', waveforms.squeeze(dim=1).cpu(), 22050)

	if True:
		# Intialize TTS (fastspeech2) and Vocoder (HiFiGAN).
		fastspeech2 = speechbrain.pretrained.FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="tmpdir_tts")
		hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder")
		#fastspeech2 = speechbrain.pretrained.FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="tmpdir_tts", run_opts={"device": "cuda"})
		#hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder", run_opts={"device": "cuda"})

		# Running the TTS.
		if True:
			input_text = "Mary had a little lamb"
		else:
			input_text = [
				"A quick brown fox jumped over the lazy dog",
				"How much wood would a woodchuck chuck?",
				"Never odd or even"
			]
		mel_output, durations, pitch, energy = fastspeech2.encode_text(input_text)  # torch.Tensor.

		# Running Vocoder (spectrogram-to-waveform).
		waveforms = hifi_gan.decode_batch(mel_output)  # torch.Tensor.

		# Save the waverform.
		torchaudio.save('./speechbrain_tts_fastspeech2.wav', waveforms.squeeze(dim=1).cpu(), 16000)

# REF [site] >> https://huggingface.co/tensorspeech
def tensor_speech_tts_example():
	# Models:
	#	tensorspeech/tts-tacotron2-ljspeech-en.
	#	tensorspeech/tts-tacotron2-kss-ko.
	#	tensorspeech/tts-tacotron2-baker-ch.
	#	tensorspeech/tts-tacotron2-thorsten-ger.
	#	tensorspeech/tts-tacotron2-synpaflex-fr.
	#
	#	tensorspeech/tts-fastspeech-ljspeech-en.
	#	tensorspeech/tts-fastspeech2-ljspeech-en.
	#	tensorspeech/tts-fastspeech2-kss-ko.
	#	tensorspeech/tts-fastspeech2-baker-ch.
	#
	#	tensorspeech/tts-mb_melgan-ljspeech-en.
	#	tensorspeech/tts-melgan-ljspeech-en.
	#	tensorspeech/tts-mb_melgan-kss-ko.
	#	tensorspeech/tts-mb_melgan-baker-ch.
	#	tensorspeech/tts-mb_melgan-thorsten-ger.
	#	tensorspeech/tts-mb_melgan-synpaflex-fr.

	# Install:
	#	pip install TensorFlowTTS

	import tensorflow as tf
	import tensorflow_tts
	import soundfile as sf

	if True:
		if True:
			# English.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
			tacotron2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
			mb_melgan = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

			text = "This is a demo to show how to use our model to generate mel spectrogram from raw text."
		else:
			# Korean.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
			tacotron2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
			mb_melgan = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")

			text = "신은 우리의 수학 문제에는 관심이 없다. 신은 다만 경험적으로 통합할 뿐이다."

		input_ids = processor.text_to_sequence(text)

		# Tacotron2 inference (text-to-mel).
		decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
			input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
			input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
			speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
		)

		#-----
		# MelGAN inference (mel-to-wav).
		audio = mb_melgan.inference(mel_outputs)[0, :, 0]

		# Save to file.
		sf.write('./tensorspeech_tts_tacotron2.wav', audio, 22050, "PCM_16")

	if True:
		if True:
			# English.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
			fastspeech2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

			text = "How are you?"
		else:
			# Korean.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
			fastspeech2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")

			text = "신은 우리의 수학 문제에는 관심이 없다. 신은 다만 경험적으로 통합할 뿐이다."

		input_ids = processor.text_to_sequence(text)

		# FastSpeech2 inference.
		mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
			input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
			speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
			speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
			f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
			energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
		)

# REF [site] >> https://huggingface.co/microsoft
def microsoft_voice_conversion_example():
	import numpy as np
	import datasets
	import soundfile as sf

	dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
	dataset = dataset.sort("id")
	sampling_rate = dataset.features["audio"].sampling_rate
	example_speech = dataset[0]["audio"]["array"]

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
	model = transformers.SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
	vocoder = transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

	inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

	# Load xvector containing speaker's voice characteristics from a file.
	speaker_embeddings = np.load("./xvector_speaker_embedding.npy")
	speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

	speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

	sf.write("./speecht5_vc.wav", speech.numpy(), samplerate=16000)

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/time_series_transformer
#	https://huggingface.co/blog/time-series-transformers
def time_series_transformer_example():
	if False:
		# Initializing a Time Series Transformer configuration with 12 time steps for prediction
		configuration = transformers.TimeSeriesTransformerConfig(prediction_length=12)

		# Randomly initializing a model (with random weights) from the configuration
		model = transformers.TimeSeriesTransformerModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		import huggingface_hub

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		batch = torch.load(file)

		model = transformers.TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

		# During training, one provides both past and future values as well as possible additional features
		outputs = model(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_values=batch["future_values"],
			future_time_features=batch["future_time_features"],
		)

		last_hidden_state = outputs.last_hidden_state
		print(f"{last_hidden_state.shape=}.")

	if True:
		import huggingface_hub

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		batch = torch.load(file)

		model = transformers.TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

		# During training, one provides both past and future values as well as possible additional features
		outputs = model(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_values=batch["future_values"],
			future_time_features=batch["future_time_features"],
		)

		loss = outputs.loss
		loss.backward()

		# During inference, one only provides past values as well as possible additional features
		# and the model autoregressively generates future values
		outputs = model.generate(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_time_features=batch["future_time_features"],
		)

		mean_prediction = outputs.sequences.mean(dim=1)
		std_prediction = outputs.sequences.std(dim=1)  # ?

		#print(f"{model.config=}.")
		print(f"{model.config.loss=}.")  # Negative log likelihood (NLL).
		print(f"{model.config.distribution_output=}.")  # Student's t-distribution
		print(f"{model.config.prediction_length=}.")
		print(f"{outputs.sequences.shape=}.")  # (batch size, number of samples, prediction length)
		print(f"{mean_prediction.shape=}.")
		print(f"{std_prediction.shape=}.")

		#-----
		# REF [site] >> https://huggingface.co/blog/time-series-transformers

		import numpy as np
		import evaluate

		forecasts = outputs.sequences
		forecast_median = np.median(forecasts, 1)

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		#file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="val-batch.pt", repo_type="dataset")
		batch = torch.load(file)
		assert len(forecast_median) == len(batch["future_values"]) == len(batch["past_values"])
		batch_size = len(forecast_median)

		#-----
		mase_metric = evaluate.load("evaluate-metric/mase")
		mape_metric = evaluate.load("evaluate-metric/mape")
		smape_metric = evaluate.load("evaluate-metric/smape")

		mase_metrics, mape_metrics, smape_metrics = [], [], []
		for idx in range(batch_size):
			mase = mase_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
				training=batch["past_values"][idx].numpy(),
				#periodicity=gluonts.time_feature.get_seasonality(freq),
			)
			mase_metrics.append(mase["mase"])

			mape = mape_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
			)
			mape_metrics.append(mape["mape"])

			smape = smape_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
			)
			smape_metrics.append(smape["smape"])
		print(f"MASE = {np.mean(mase_metrics)}, MAPE = {np.mean(mape_metrics)}, sMAPE = {np.mean(smape_metrics)}.")

		#-----
		mase_metric = evaluate.load("evaluate-metric/mase", "multilist")
		mape_metric = evaluate.load("evaluate-metric/mape", "multilist")
		smape_metric = evaluate.load("evaluate-metric/smape", "multilist")

		mase = mase_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			training=batch["past_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
			#periodicity=gluonts.time_feature.get_seasonality(freq),
		)
		mape = mape_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
		)
		smape = smape_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
		)
		print(f'MASE = {np.mean(mase["mase"])}, MAPE = {np.mean(mape["mape"])}, sMAPE = {np.mean(smape["smape"])}.')

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/decision_transformer
#	https://huggingface.co/edbeeching
#	https://github.com/huggingface/transformers/blob/main/examples/research_projects/decision_transformer/run_decision_transformer.py
def decision_transformer_example():
	# Models:
	#	edbeeching/decision-transformer-gym-halfcheetah-medium.
	#	edbeeching/decision-transformer-gym-halfcheetah-medium-replay.
	#	edbeeching/decision-transformer-gym-halfcheetah-expert.
	#	edbeeching/decision-transformer-gym-hopper-medium.
	#	edbeeching/decision-transformer-gym-hopper-medium-replay.
	#	edbeeching/decision-transformer-gym-hopper-expert.
	#	edbeeching/decision-transformer-gym-hopper-expert-new.
	#	edbeeching/decision-transformer-gym-walker2d-medium.
	#	edbeeching/decision-transformer-gym-walker2d-medium-replay.

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		# Initializing a DecisionTransformer configuration.
		configuration = transformers.DecisionTransformerConfig()

		# Initializing a model (with random weights) from the configuration.
		model = transformers.DecisionTransformerModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if True:
		import gym

		model = transformers.DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
		model = model.to(device)
		model.eval()  # Evaluation.

		env = gym.make("Hopper-v3")
		state_dim = env.observation_space.shape[0]
		act_dim = env.action_space.shape[0]

		state = env.reset()
		states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
		actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
		rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
		target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
		timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
		attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

		# Forward pass.
		with torch.no_grad():
			state_preds, action_preds, return_preds = model(
				states=states,
				actions=actions,
				rewards=rewards,
				returns_to_go=target_return,
				timesteps=timesteps,
				attention_mask=attention_mask,
				return_dict=False,
			)

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/trajectory_transformer
def trajectory_transformer_example():
	# Models:
	#	CarlCochet/trajectory-transformer-ant-medium-v2
	#	CarlCochet/trajectory-transformer-ant-medium-replay-v2
	#	CarlCochet/trajectory-transformer-ant-medium-expert-v2
	#	CarlCochet/trajectory-transformer-ant-expert-v2.
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-replay-v2
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-v2.
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-expert-v2
	#	CarlCochet/trajectory-transformer-halfcheetah-expert-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-replay-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-expert-v2
	#	CarlCochet/trajectory-transformer-hopper-expert-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-replay-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-expert-v2
	#	CarlCochet/trajectory-transformer-walker2d-expert-v2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		# Initializing a TrajectoryTransformer CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
		configuration = transformers.TrajectoryTransformerConfig()

		# Initializing a model (with random weights) from the CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
		model = transformers.TrajectoryTransformerModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		model = TrajectoryTransformerModel.from_pretrained("CarlCochet/trajectory-transformer-halfcheetah-medium-v2")
		model.to(device)
		model.eval()

		observations_dim, action_dim, batch_size = 17, 6, 256
		seq_length = observations_dim + action_dim + 1

		trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)
		targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

		outputs = model(
			trajectories,
			targets=targets,
			use_cache=True,
			output_attentions=True,
			output_hidden_states=True,
			return_dict=True,
		)

# REF [site] >> https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama/scripts
def stack_llama_example():
	raise NotImplementedError

# REF [site] >> https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py
def stack_llama_2_sft_llama2_example():
	raise NotImplementedError

# REF [site] >> https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py
def stack_llama_2_dpo_llama2_example():
	raise NotImplementedError

# REF [site] >> https://github.com/keitazoumana/Medium-Articles-Notebooks/blob/main/Train_your_LLM.ipynb
def trl_train_small_llm_example():
	import trl
	import peft
	import datasets
	import textwrap
	import matplotlib.pyplot as plt

	#-----
	# Prepare data

	train_dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")

	print(train_dataset)

	# We can get the first five rows as follows
	pandas_format = train_dataset.to_pandas()
	print(pandas_format.head())

	for index in range(3):
		print("---" * 15)
		print("Instruction: {}".format(textwrap.fill(pandas_format.iloc[index]["instruction"], width=50)))
		print("Output: {}".format(textwrap.fill(pandas_format.iloc[index]["output"], width=50)))
		print("Text: {}".format(textwrap.fill(pandas_format.iloc[index]["text"], width=50)))

	pandas_format["text_length"] = pandas_format["text"].apply(len)
	max_length = pandas_format["text_length"].max()

	plt.figure(figsize=(10, 6))
	plt.hist(pandas_format["text_length"], bins=50, alpha=0.5, color="g")
	plt.annotate("Max length: {}".format(max_length), xy=(max_length, 0), xytext=(max_length, 50), arrowprops=dict(facecolor="red", shrink=0.05))
	plt.title("Distribution of Length of Text")
	plt.xlabel("Length of Text")
	plt.ylabel("Frequency")
	plt.grid(True)
	plt.show()

	mask = pandas_format["text_length"] > 1024
	percentage = (mask.sum() / pandas_format["text_length"].count()) * 100

	print(f"The percentage of text documents with a length greater than 1024 is: {percentage}%")

	#-----
	# Build a model
	pretrained_model_name = "Salesforce/xgen-7b-8k-base"

	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
	model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)  # ~27.6GB

	#-----
	# Training

	training_args = transformers.TrainingArguments(
		output_dir="xgen-7b-8k-base-fine-tuned",
		per_device_train_batch_size=4,
		optim="adamw_torch",
		logging_steps=80,
		learning_rate=2e-4,
		warmup_ratio=0.1,
		lr_scheduler_type="linear",
		num_train_epochs=1,
		save_strategy="epoch",
	)

	peft_lora_config = peft.LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		task_type="CAUSAL_LM",
	)

	sft_trainer = trl.SFTTrainer(
		model=model,
		train_dataset=train_dataset,
		dataset_text_field="text",
		max_seq_length=1024,
		tokenizer=tokenizer,
		args=training_args,
		packing=True,
		peft_config=peft_lora_config,
	)

	tokenizer.pad_token = tokenizer.eos_token
	model.resize_token_embeddings(len(tokenizer))
	model = peft.prepare_model_for_int8_training(model)
	model = peft.get_peft_model(model, lora_peft_config)

	sft_trainer.train()

	sft_trainer.save_model(training_args.output_dir)
	sft_trainer.save_state()
	output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
	sft_trainer.model.save_pretrained(output_dir)

# REF [site] >> https://github.com/davidkim205/komt/blob/main/dpo_train.py
def dpo_train_example():
	import os, typing
	import peft
	import trl
	import datasets

	def get_stack_exchange_paired(
		dataset_name: str = "",
		sanity_check: bool = False,
		cache_dir: str = None,
		num_proc=24,
	) -> datasets.Dataset:
		"""Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

		The dataset is converted to a dictionary with the following structure:
		{
			"prompt": List[str],
			"chosen": List[str],
			"rejected": List[str],
		}

		"""
		dataset = datasets.load_dataset(
			dataset_name,
			split="train",
			cache_dir=cache_dir,
			#data_dir=data_dir,
		)
		original_columns = dataset.column_names

		if sanity_check:
			dataset = dataset.select(range(min(len(dataset), 100)))

		print("dataset length =", len(dataset))
		def return_prompt_and_responses(samples) -> typing.Dict[str, str]:
			return {
				"prompt": ["[INST]" + question + "[/INST]" for question in samples["prompt"]],
				"chosen": samples["chosen"],
				"rejected": samples["rejected"],
			}

		return dataset.map(
			return_prompt_and_responses,
			batched=True,
			num_proc=num_proc,
			remove_columns=original_columns,
		)

	model_name_or_path: typing.Optional[str] = "davidkim205/komt-mistral-7b-v1"
	dataset_name: typing.Optional[str] = "maywell/ko_Ultrafeedback_binarized"
	learning_rate: typing.Optional[float] = 5e-4  # Optimizer learning rate
	lr_scheduler_type: typing.Optional[str] = "cosine"  # The lr scheduler type
	warmup_steps: typing.Optional[int] = 100  # The number of warmup steps
	#weight_decay: typing.Optional[float] = 0.05  # The weight decay
	optimizer_type: typing.Optional[str] = "paged_adamw_32bit"  # The optimizer type
	per_device_train_batch_size: typing.Optional[int] = 8  # Train batch size per device
	per_device_eval_batch_size: typing.Optional[int] = 2  # Eval batch size per device
	gradient_accumulation_steps: typing.Optional[int] = 4  # The number of gradient accumulation steps
	gradient_checkpointing: typing.Optional[bool] = True  # Whether to use gradient checkpointing
	max_prompt_length: typing.Optional[int] = 512  # The maximum prompt length
	max_length: typing.Optional[int] = 1024  # The maximum sequence length
	max_steps: typing.Optional[int] = 1000  # Max number of training steps
	logging_steps: typing.Optional[int] = 10  # The logging frequency
	save_steps: typing.Optional[int] = 300  # The saving frequency
	eval_steps: typing.Optional[int] = 300  # The evaluation frequency
	output_dir: typing.Optional[str] = ""  # The output directory
	sanity_check: typing.Optional[bool] = False
	report_to: typing.Optional[str] = None  # The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`, `"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. Use `"all"` to report to all integrations installed, `"none"` for no integrations
	ignore_bias_buffers: typing.Optional[bool] = False  # Debug argument for distributed training

	# Load a pretrained model
	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_name_or_path,
		#low_cpu_mem_usage=True,
		torch_dtype=torch.float16,
		load_in_4bit=True,
	)
	model.config.use_cache = False

	if ignore_bias_buffers:
		# torch distributed hack
		model._ddp_params_and_buffers_to_ignore = [
			name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
		]

	model_ref = transformers.AutoModelForCausalLM.from_pretrained(
		model_name_or_path,
		#low_cpu_mem_usage=True,
		torch_dtype=torch.float16,
		load_in_4bit=True,
	)

	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
	tokenizer.pad_token = tokenizer.eos_token

	# Load the Stack-exchange paired dataset
	train_dataset = get_stack_exchange_paired(dataset_name=dataset_name, sanity_check=sanity_check)
	train_dataset = train_dataset.filter(
		lambda x: len(x["prompt"]) + max(len(x["chosen"]), len(x["rejected"])) <= max_length
	)

	# Load evaluation dataset
	eval_dataset = get_stack_exchange_paired(dataset_name=dataset_name, sanity_check=True)
	eval_dataset = eval_dataset.filter(
		lambda x: len(x["prompt"]) + max(len(x["chosen"]), len(x["rejected"])) <= max_length
	)

	if len(output_dir) <= 1:
		model_name = model_name_or_path.replace("../", "")
		model_name = model_name.replace("..", "").replace("/", "_")
		dataset_name = dataset_name.replace("/", "_")
		batch = per_device_train_batch_size
		output_dir = f"{model_name}_{dataset_name}_lr{learning_rate}_{lr_scheduler_type}_b{batch}_step{max_steps}"
		print("Output dir:", output_dir)

	# Initialize training arguments
	training_args = transformers.TrainingArguments(
		per_device_train_batch_size=per_device_train_batch_size,
		per_device_eval_batch_size=per_device_eval_batch_size,
		max_steps=max_steps,
		logging_steps=logging_steps,
		save_steps=save_steps,
		gradient_accumulation_steps=gradient_accumulation_steps,
		gradient_checkpointing=gradient_checkpointing,
		learning_rate=learning_rate,
		evaluation_strategy="steps",
		eval_steps=eval_steps,
		output_dir=output_dir,
		report_to=report_to,
		lr_scheduler_type=lr_scheduler_type,
		warmup_steps=warmup_steps,
		optim=optimizer_type,
		bf16=True,
		remove_unused_columns=False,
		run_name="dpo_llama2",
	)

	peft_lora_config = peft.LoraConfig(
		r=8,  # The lora r parameter
		lora_alpha=16,  # The lora alpha parameter
		lora_dropout=0.05,  # The lora dropout parameter
		target_modules=[
			"q_proj",
			"v_proj",
			"k_proj",
			"out_proj",
			"fc_in",
			"fc_out",
			"wte",
		],
		bias="none",
		task_type="CAUSAL_LM",
	)

	# Initialize the DPO trainer
	dpo_trainer = trl.DPOTrainer(
		model,
		#ref_model=model_ref,  # ValueError: You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference model. Please pass `ref_model=None` in case you want to train PEFT adapters.
		ref_model=None,
		args=training_args,
		beta=0.1,  # The beta parameter for DPO loss
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		peft_config=peft_lora_config,
		max_prompt_length=max_prompt_length,
		max_length=max_length,
	)

	# Train
	dpo_trainer.train()
	dpo_trainer.save_model(training_args.output_dir)
	dpo_trainer.save_state()

	# Save
	output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
	dpo_trainer.model.save_pretrained(output_dir)

def main():
	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/transformer_test.py

	# REF [site] >> https://huggingface.co/docs/transformers/index

	#quick_tour()

	#transformers_test()
	#tokenizer_test()

	#--------------------
	# Pipeline.
	#	https://huggingface.co/docs/transformers/main/en/pipeline_tutorial

	#pipeline_example()

	#question_answering_example()

	#korean_fill_mask_example()
	#korean_table_question_answering_example()  # Not correctly working.

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

	#--------------------
	# Data and model parallelism.

	# Efficient Training on Multiple GPUs:
	#	https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many

	# Model Parallelism:
	#	https://huggingface.co/docs/transformers/v4.19.4/en/parallelism

	# Refer to llama2_example().

	#--------------------
	# Model.

	#encoder_decoder_example()
	#perceiver_example()  # Perceiver & Perceiver IO.

	#--------------------
	# Language.

	#-----
	# GPT.

	#gpt2_example()
	#sentence_completion_model_using_gpt2_example()
	#conditional_text_generation_using_gpt2_example()  # Not yet implemented.

	#eleuther_ai_gpt_test()  # gpt-j, gpt-neo, & gpt-neox.
	#skt_gpt_test()  # KoGPT2.
	#kakao_brain_gpt_test()  # KoGPT.

	#gpt4all_example()  # Not correctly working.

	#-----
	# BERT.

	#bert_example()
	#masked_language_modeling_for_bert_example()

	#sequence_classification_using_bert()

	#korean_bert_example()  # BERT multilingual & KoBERT.
	#skt_bert_test()  # KoBERT. Not yet tested.
	#klue_bert_test()  # Not yet completed.

	#-----
	# T5.
	#	T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.
	#	T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g., for translation: translate English to German: ..., for summarization: summarize: ....

	#t5_example()
	#flan_t5_example()

	#-----
	#palm_example()  # PaLM + RLHF.
	#bloom_example()  # BLOOM.

	#opt_example()  # OPT.
	#galactica_example()  # Galactica.

	#llama_example()  # LLaMA.
	#llama2_example()  # Llama 2. Model parallelism.
	#llama3_example()  # Llama 3.
	#open_llama_example()  # OpenLLaMA.

	#megatron_example()  # Megatron-LM.
	#mpt_example()  # MPT.

	#falcon_example()  # Falcon.
	#yi_example()  # Yi-6B & Yi-34B.

	#orca_example()  # ORCA-2.

	#mistral_example()  # Mistral-7B.
	#mixtral_example()  # Mixtral-8x7B.
	#zephyr_example()  # Zephyr-7B = Mistral-7B + DPO. Not yet tested.
	#gemma_example()  # Gemma. Not yet tested.

	#rag_example()  # Retrieval-augmented generation (RAG).

	#-----
	# Code.

	#codebert_example()  # CodeBERT.
	#codeberta_example()  # CodeBERTa.
	#codet5_example()  # CodeT5.
	#codet5p_example()  # CodeT5+.
	#codegen_example()  # CodeGen.
	#codegen2_example()  # CodeGen2.
	#codegen25_example()  # CodeGen2.5.
	#codeparrot_example()  # CodeParrot.
	#code_llama_example()  # Code Llama.
	#star_coder_example()  # StarCoder. Not yet tested.
	#replit_example()  # Replit. Not yet tested.
	#phi_example()  # phi-1 & phi-1.5.

	#--------------------
	# Vision.

	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/vit_test.py

	#vit_example()  # ViT.
	#deit_example()  # DeiT.

	#dino_example()  # DINO & DINOv2. Not yet tested.
	#dpt_example()  # Dense Prediction Transformer (DPT). Not yet tested.

	#--------------------
	# Vision and language.

	#vilt_example()  # ViLT.
	#beit_example()  # BEiT.

	#clip_example()  # CLIP.
	#align_example()  # ALIGN.
	#git_example()  # GIT.
	#blip_example()  # BLIP.
	#openflamingo_example()  # OpenFlamingo.

	#--------------------
	# Vision and audio.

	#tvlt_example()  # TVLT.

	#--------------------
	# Multimodal.

	#kosmos_example()  # Kosmos-2.

	#--------------------
	# Document.

	# LayoutLM.
	#	Required libraries: tesseract, detectron2.
	#layoutlmv2_example()  # LayoutLMv2.
	#layoutlmv3_example()  # LayoutLMv3.

	# Donut.
	#donut_example()
	#donut_invoice_test()  # Not yet completed.

	#-----
	# Table.

	#table_transformer_example()  # Table Transformer (TATR).
	#tapex_example()  # TaPEx.

	#--------------------
	# Text.

	#trocr_example()  # TrOCR.

	#--------------------
	# Speech.

	#-----
	# Speech recognition.

	#microsoft_asr_example()  # SpeechT5.
	#nvidia_asr_example()  # Conformer, Citrinet, Parakeet, Canary.
	#openai_asr_example()  # Whisper.
	#speech_brain_asr_example()  # Whisper. Error.

	#-----
	# Speech synthesis.

	#microsoft_tts_example()  # SpeechT5.
	#nvidia_tts_example()  # FastPitch + HiFiGAN.
	#speech_brain_tts_example()  # Tacotron / FastSpeech + HiFiGAN.
	#tensor_speech_tts_example()  # Tacotron / FastSpeech + MelGAN.

	#-----
	# Speech-to-speech.

	#microsoft_voice_conversion_example()  # SpeechT5.

	#--------------------
	# Sequence.

	#time_series_transformer_example()  # Probabilistic time series transformer.

	#--------------------
	# Learning theory.

	#decision_transformer_example()  # Decision transformer. Not yet tested.
	#trajectory_transformer_example()  # Trajectory transformer. Not yet tested.

	#--------------------
	# Transformer Reinforcement Learning (TRL).
	#	https://github.com/huggingface/trl
	#	https://huggingface.co/docs/trl/index
	#
	#	Supervised Fine-tuning (SFT).
	#	Reward Modeling (RM).
	#	Proximal Policy Optimization (PPO).
	#	Direct Preference Optimization (DPO).

	#stack_llama_example()  # Reinforcement Learning from Human Feedback (RLHF). Not yet implemented.
	#stack_llama_2_sft_llama2_example()  # Not yet implemented.
	#stack_llama_2_dpo_llama2_example()  # Not yet implemented.

	#trl_train_small_llm_example()  # XGen-7B + SFT. Not yet tested.
	#dpo_train_example()  # Mistral-7B + DPO.

	#--------------------
	# Inference engine.
	#	https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407

	#--------------------
	# Fine-tune LLMs.
	#	${SWR_HOME}/test/language_processing/llm_fine_tuning_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
