#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/huggingface/transformers
#	https://huggingface.co/transformers/
#	https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d

import time
import torch
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

def tokenizer_test():
	import transformers

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

		# Install.
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

		# Install.
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

		# Install.
		#	pip install bitsandbytes accelerate

		from transformers import T5Tokenizer, T5ForConditionalGeneration

		tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
		model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', device_map='auto', load_in_8bit=True)

		input_text = 'translate English to German: How old are you?'
		input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')

		outputs = model.generate(input_ids)
		print(tokenizer.decode(outputs[0]))

# REF [site] >>
#	https://huggingface.co/bigscience
#	https://huggingface.co/docs/transformers/model_doc/bloom
def bloom_example():
	# BigScience Language Open-science Open-access Multilingual (BLOOM) language model.
	#	Modified from Megatron-LM GPT2 (see paper, BLOOM Megatron code).
	#	Decoder-only architecture
	#	Layer normalization applied to word embeddings layer (StableEmbedding; see code, paper).
	#	ALiBi positional encodings (see paper), with GeLU activation functions.
	#	176, 247, 271, 424 parameters:
	#		3, 596, 615, 680 embedding parameters.
	#		70 layers, 112 attention heads.
	#		Hidden layers are 14336-dimensional.
	#		Sequence length of 2048 tokens used (see BLOOM tokenizer, tokenizer description).
	#	Objective Function: Cross Entropy with mean reduction.

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

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_id = logits.argmax().item()

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

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_token_class_ids = logits.argmax(-1)

		# Note that tokens are classified rather then input words which means that there might be more predicted token classes than words.
		# Multiple token classes might account for the same word.
		predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

		labels = predicted_token_class_ids
		loss = model(**inputs, labels=labels).loss

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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vit
def vit_example():
	import requests
	import torch
	from PIL import Image
	from transformers import ViTImageProcessor, ViTModel, ViTForMaskedImageModeling
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
	model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

	#-----
	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
	model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

	num_patches = (model.config.image_size // model.config.patch_size)**2
	pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
	# Create random boolean mask of shape (batch_size, num_patches).
	bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

	outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

	loss, reconstructed_pixel_values = outputs.loss, outputs.logits
	print(reconstructed_pixel_values.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vit
def vit_masked_image_modeling_example():
	import requests
	import torch
	from PIL import Image
	from transformers import ViTImageProcessor, ViTForMaskedImageModeling

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
	model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

	num_patches = (model.config.image_size // model.config.patch_size)**2
	pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
	# Create random boolean mask of shape (batch_size, num_patches).
	bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

	outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

	loss, reconstructed_pixel_values = outputs.loss, outputs.logits
	print(reconstructed_pixel_values.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vit
def vit_image_classification_example():
	import torch
	from transformers import ViTImageProcessor, ViTForImageClassification
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
	model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		logits = model(**inputs).logits

	# Model predicts one of the 1000 ImageNet classes.
	predicted_label = logits.argmax(-1).item()
	print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/deit
def deit_example():
	import torch
	from transformers import DeiTImageProcessor, DeiTModel
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
	model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/deit
def deit_masked_image_modeling_example():
	import requests
	import torch
	from PIL import Image
	from transformers import DeiTImageProcessor, DeiTForMaskedImageModeling

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
	model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

	num_patches = (model.config.image_size // model.config.patch_size) ** 2
	pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
	# Create random boolean mask of shape (batch_size, num_patches).
	bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

	outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

	loss, reconstructed_pixel_values = outputs.loss, outputs.logits
	print(reconstructed_pixel_values.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/deit
def deit_image_classification_example():
	import requests
	import torch
	from PIL import Image
	from transformers import DeiTImageProcessor, DeiTForImageClassification

	torch.manual_seed(3)

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	# Note: we are loading a DeiTForImageClassificationWithTeacher from the hub here,
	# so the head will be randomly initialized, hence the predictions will be random.
	image_processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
	model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

	inputs = image_processor(images=image, return_tensors="pt")
	outputs = model(**inputs)

	logits = outputs.logits
	# Model predicts one of the 1000 ImageNet classes.
	predicted_class_idx = logits.argmax(-1).item()
	print("Predicted class: {}.".format(model.config.id2label[predicted_class_idx]))

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/deit
def deit_image_classification_with_teacher_example():
	import torch
	from transformers import DeiTImageProcessor, DeiTForImageClassificationWithTeacher
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
	model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		logits = model(**inputs).logits

	# Model predicts one of the 1000 ImageNet classes.
	predicted_label = logits.argmax(-1).item()
	print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/beit
def beit_example():
	import torch
	from transformers import BeitImageProcessor, BeitModel
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
	model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/beit
def beit_masked_image_modeling_example():
	import requests
	import torch
	from PIL import Image
	from transformers import BeitImageProcessor, BeitForMaskedImageModeling

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
	model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

	num_patches = (model.config.image_size // model.config.patch_size)**2
	pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
	# Create random boolean mask of shape (batch_size, num_patches).
	bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

	outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

	loss, logits = outputs.loss, outputs.logits
	print(logits.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/beit
def beit_image_classification_example():
	import torch
	from transformers import BeitImageProcessor, BeitForImageClassification
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
	model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		logits = model(**inputs).logits

	# Model predicts one of the 1000 ImageNet classes.
	predicted_label = logits.argmax(-1).item()
	print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/beit
def beit_semantic_segmentation_example():
	import requests
	from PIL import Image
	from transformers import AutoImageProcessor, BeitForSemanticSegmentation

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
	model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

	inputs = image_processor(images=image, return_tensors="pt")
	outputs = model(**inputs)

	# Logits are of shape (batch_size, num_labels, height, width).
	logits = outputs.logits
	print(logits.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_example():
	import requests
	from PIL import Image
	from transformers import ViltProcessor, ViltModel

	# Prepare image and text.
	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)
	text = "hello world"

	processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
	model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

	inputs = processor(image, text, return_tensors="pt")
	outputs = model(**inputs)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_masked_lm_example():
	import re, requests
	from PIL import Image
	import torch
	from transformers import ViltProcessor, ViltForMaskedLM

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)
	text = "a bunch of [MASK] laying on a [MASK]."

	processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
	model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

	# Prepare inputs.
	encoding = processor(image, text, return_tensors="pt")

	# Forward pass.
	outputs = model(**encoding)

	tl = len(re.findall("\[MASK\]", text))
	inferred_token = [text]

	# Gradually fill in the MASK tokens, one by one.
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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_question_answering_example():
	import requests
	from PIL import Image
	from transformers import ViltProcessor, ViltForQuestionAnswering

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)
	text = "How many cats are there?"

	processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
	model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

	# Prepare inputs.
	encoding = processor(image, text, return_tensors="pt")

	# Forward pass.
	outputs = model(**encoding)

	logits = outputs.logits
	idx = logits.argmax(-1).item()
	print("Predicted answer:", model.config.id2label[idx])

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_images_and_text_classification_example():
	import requests
	from PIL import Image
	from transformers import ViltProcessor, ViltForImagesAndTextClassification

	image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
	image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg", stream=True).raw)
	text = "The left image contains twice the number of dogs as the right image."

	processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
	model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

	# Prepare inputs.
	encoding = processor([image1, image2], text, return_tensors="pt")

	# Forward pass.
	outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))

	logits = outputs.logits
	idx = logits.argmax(-1).item()
	print("Predicted answer: {}.".format(model.config.id2label[idx]))

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/vilt
def vilt_image_and_text_retrieval_example():
	import requests
	from PIL import Image
	from transformers import ViltProcessor, ViltForImageAndTextRetrieval

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)
	texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

	processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
	model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

	# Forward pass.
	scores = dict()
	for text in texts:
		# Prepare inputs.
		encoding = processor(image, text, return_tensors="pt")
		outputs = model(**encoding)
		scores[text] = outputs.logits[0, :].item()

	print(scores)

# REF [site] >>
#	https://huggingface.co/openai
#	https://huggingface.co/docs/transformers/main/model_doc/clip
def clip_example():
	# Models:
	#	openai/clip-vit-base-patch16.
	#	openai/clip-vit-base-patch32: ~605MB.
	#	openai/clip-vit-large-patch14.
	#	openai/clip-vit-large-patch14-336.

	import requests
	from PIL import Image
	import torch
	import transformers

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

	import requests
	from PIL import Image
	import torch
	import transformers

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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv2
def layoutlmv2_example():
	#import torch
	from PIL import Image
	from transformers import LayoutLMv2Processor, LayoutLMv2Model, set_seed
	from datasets import load_dataset

	set_seed(88)

	processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
	model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")

	dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
	image_path = dataset["test"][0]["file"]
	image = Image.open(image_path).convert("RGB")
	encoding = processor(image, return_tensors="pt")

	outputs = model(**encoding)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv2
def layoutlmv2_sequence_classification_example():
	import torch
	from PIL import Image
	from transformers import LayoutLMv2Processor, LayoutLMv2ForSequenceClassification, set_seed
	from datasets import load_dataset

	set_seed(88)

	dataset = load_dataset("rvl_cdip", split="train", streaming=True)
	data = next(iter(dataset))
	image = data["image"].convert("RGB")

	processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
	model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes)

	encoding = processor(image, return_tensors="pt")
	sequence_label = torch.tensor([data["label"]])

	outputs = model(**encoding, labels=sequence_label)

	loss, logits = outputs.loss, outputs.logits
	predicted_idx = logits.argmax(dim=-1).item()
	predicted_answer = dataset.info.features["label"].names[predicted_idx]
	print(f"Predicted: index = {predicted_idx}, answer = {predicted_answer}.")

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv2
def layoutlmv2_token_classification_example():
	from PIL import Image
	from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification, set_seed
	from datasets import load_dataset

	set_seed(88)

	datasets = load_dataset("nielsr/funsd", split="test")
	labels = datasets.features["ner_tags"].feature.names
	id2label = {v: k for v, k in enumerate(labels)}

	processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
	model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=len(labels))

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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv2
def layoutlmv2_question_answering_example():
	import torch
	from PIL import Image
	from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering, set_seed
	from datasets import load_dataset

	set_seed(88)

	processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
	model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

	dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
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
	from transformers import AutoProcessor, AutoModel
	from datasets import load_dataset

	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
	model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

	dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
	example = dataset[0]
	image = example["image"]
	words = example["tokens"]
	boxes = example["bboxes"]
	encoding = processor(image, words, boxes=boxes, return_tensors="pt")

	outputs = model(**encoding)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv3
def layoutlmv3_sequence_classification_example():
	import torch
	from transformers import AutoProcessor, AutoModelForSequenceClassification
	from datasets import load_dataset

	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
	model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

	dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv3
def layoutlmv3_token_classification_example():
	from transformers import AutoProcessor, AutoModelForTokenClassification
	from datasets import load_dataset

	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
	model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

	dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/layoutlmv3
def layoutlmv3_question_answering_example():
	import torch
	from transformers import AutoProcessor, AutoModelForQuestionAnswering
	from datasets import load_dataset

	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
	model = AutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

	dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
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
	import torch
	from transformers import AutoFeatureExtractor, DonutSwinModel
	from datasets import load_dataset

	dataset = load_dataset("huggingface/cats-image")
	image = dataset["test"]["image"][0]

	feature_extractor = AutoFeatureExtractor.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")
	model = DonutSwinModel.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")

	inputs = feature_extractor(image, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs)

	last_hidden_states = outputs.last_hidden_state
	print(last_hidden_states.shape)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/donut
def donut_document_image_classification_example():
	import re
	import torch
	from transformers import DonutProcessor, VisionEncoderDecoderModel
	from datasets import load_dataset

	device = "cuda" if torch.cuda.is_available() else "cpu"

	processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
	model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
	model.to(device)

	# Load document image.
	dataset = load_dataset("hf-internal-testing/example-documents", split="test")
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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/donut
def donut_document_document_parsing_example():
	import re
	import torch
	from transformers import DonutProcessor, VisionEncoderDecoderModel
	from datasets import load_dataset

	device = "cuda" if torch.cuda.is_available() else "cpu"

	processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
	model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
	model.to(device)

	# Load document image.
	dataset = load_dataset("hf-internal-testing/example-documents", split="test")
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

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/donut
def donut_document_visual_question_answering_example():
	import re
	import torch
	from transformers import DonutProcessor, VisionEncoderDecoderModel
	from datasets import load_dataset

	device = "cuda" if torch.cuda.is_available() else "cpu"

	processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
	model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
	model.to(device)

	# Load document image from the DocVQA dataset.
	dataset = load_dataset("hf-internal-testing/example-documents", split="test")
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
	from PIL import Image
	model = DonutModel.from_pretrained("./custom-fine-tuned-model")

	prediction = model.inference(
		image=Image.open("./example-invoice.jpeg"), prompt="<s_dataset-donut-generated>"
	)["predictions"][0]

	print(prediction)

def whisper_example():
	# Models:
	#	openai/whisper-tiny.
	#	openai/whisper-base.
	#	openai/whisper-small.
	#	openai/whisper-medium.
	#	openai/whisper-large: ~6.17GB.
	#	openai/whisper-large-v2.

	import transformers
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

		import torch
		import evaluate

		device = "cuda" if torch.cuda.is_available() else "cpu"

		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)

		def map_to_pred(batch):
			audio = batch["audio"]
			input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
			batch["reference"] = processor.tokenizer._normalize(batch["text"])

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

		import torch

		device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def main():
	# REF [site] >> https://huggingface.co/docs/transformers/index

	#quick_tour()
	#tokenizer_test()

	#--------------------
	# Pipeline.

	#pipeline_example()

	#question_answering_example()

	#korean_fill_mask_example()
	#korean_table_question_answering_example()  # Not correctly working.

	#--------------------
	# GPT.

	#gpt2_example()
	#sentence_completion_model_using_gpt2_example()
	#conditional_text_generation_using_gpt2_example()  # Not yet implemented.

	eleuther_ai_gpt_test()  # gpt-j, gpt-neo, & gpt-neox.
	#skt_gpt_test()  # KoGPT2.
	#kakao_brain_gpt_test()  # KoGPT.

	#--------------------
	# BERT.

	#bert_example()
	#masked_language_modeling_for_bert_example()

	#sequence_classification_using_bert()

	#korean_bert_example()  # BERT multilingual & KoBERT.
	#skt_bert_test()  # KoBERT. Not yet tested.
	#klue_bert_test()  # Not yet completed.

	#--------------------
	# T5.
	#	T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.
	#	T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g., for translation: translate English to German: ..., for summarization: summarize: ....

	#t5_example()
	#flan_t5_example()

	#--------------------
	# BLOOM.

	#bloom_example()

	#--------------------
	#encoder_decoder_example()

	#--------------------
	# Vision.

	#vit_example()
	#vit_masked_image_modeling_example()
	#vit_image_classification_example()

	#deit_example()
	#deit_masked_image_modeling_example()
	#deit_image_classification_example()
	#deit_image_classification_with_teacher_example()

	#beit_example()
	#beit_masked_image_modeling_example()
	#beit_image_classification_example()
	#beit_semantic_segmentation_example()

	#--------------------
	# Vision and language.

	#vilt_example()
	#vilt_masked_lm_example()
	#vilt_question_answering_example()
	#vilt_images_and_text_classification_example()
	#vilt_image_and_text_retrieval_example()

	#clip_example()
	#align_example()

	#--------------------
	# Document processing.

	# LayoutLM requires libraries: tesseract, detectron2.
	#layoutlmv2_example()
	#layoutlmv2_sequence_classification_example()
	#layoutlmv2_token_classification_example()
	#layoutlmv2_question_answering_example()

	#layoutlmv3_example()
	#layoutlmv3_sequence_classification_example()
	#layoutlmv3_token_classification_example()
	#layoutlmv3_question_answering_example()

	#donut_example()
	#donut_document_image_classification_example()
	#donut_document_document_parsing_example()
	#donut_document_visual_question_answering_example()

	#donut_invoice_test()  # Not yet completed.

	#--------------------
	# Speech recognition.

	#whisper_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
