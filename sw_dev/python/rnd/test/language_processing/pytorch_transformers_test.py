#!/usr/bin/env python
# coding: UTF-8

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d

import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def gpt2_example():
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

def main():
	#sentence_completion_model_using_gpt2_example()
	conditional_text_teneration_using_gpt2_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
