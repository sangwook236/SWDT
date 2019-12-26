#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d

import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def sentence_completion_model_using_gpt2_example():
	# Load pre-trained model tokenizer (vocabulary).
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

	# Encode a text inputs.
	text = 'What is the fastest car in the'
	indexed_tokens = tokenizer.encode(text)

	# Convert indexed tokens in a PyTorch tensor.
	tokens_tensor = torch.tensor([indexed_tokens])

	# Load pre-trained model (weights).
	model = GPT2LMHeadModel.from_pretrained('gpt2')

	# Set the model in evaluation mode to deactivate the DropOut modules.
	model.eval()

	# If you have a GPU, put everything on cuda.
	tokens_tensor = tokens_tensor.to('cuda')
	model.to('cuda')

	# Predict all tokens.
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]

	# Get the predicted next sub-word.
	predicted_index = torch.argmax(predictions[0, -1, :]).item()
	predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

	# Print the predicted word.
	print('Predicted text =', predicted_text)

# REF [site] >>
#	https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
#		python pytorch-transformers/examples/run_generation.py --model_type=gpt2 --length=100 --model_name_or_path=gpt2
#	https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def conditional_text_teneration_using_gpt2_example():
	raise NotImplementedError

# REF [site] >> https://www.analyticsvidhya.com/blog/2019/07/pytorch-transformers-nlp-python/?utm_source=blog&utm_medium=openai-gpt2-text-generator-python
def masked_language_modeling_for_BERT_example():
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

	# Load pre-trained model (weights).
	model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	model.eval()

	# If you have a GPU, put everything on cuda.
	tokens_tensor = tokens_tensor.to('cuda')
	segments_tensors = segments_tensors.to('cuda')
	model.to('cuda')

	# Predict all tokens.
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
		predictions = outputs[0]

	# Confirm we were able to predict 'henson'.
	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	assert predicted_token == 'henson'
	print('Predicted token is:', predicted_token)

def main():
	#sentence_completion_model_using_gpt2_example()
	#conditional_text_teneration_using_gpt2_example()

	masked_language_modeling_for_BERT_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
