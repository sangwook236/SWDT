#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
def detr_positional_encoding_test():
	raise NotImplementedError

	#pe = PositionEmbeddingSine(num_pos_feats=64, temperature=10000, normalize=False, scale=None)
	#pe = PositionEmbeddingLearned(num_pos_feats=256)

# REF [site] >>
#	https://nlp.seas.harvard.edu/2018/04/03/attention.html
#	https://github.com/fengxinjie/Transformer-OCR
def harvard_nlp_transformer_test():
	raise NotImplementedError

def main():
	# Standard transformer model.
	#	Encoder-decoder transformer model = transformer model architecture in an encoder-decoder setup.
	# Encoder-only or decoder-only transformer model.
	#	Encoder-only transformer model = transformer model architecture in an encoder-only setup.
	#	Decoder-only transformer model = transformer model architecture in a decoder-only setup.

	# NOTE [info] >> The difference between encoder-only transformer and decoder-only transformer models.
	#	Both models use only the left part of the standard transformer model.
	#	Encoder-only transformer model uses the multi-head self-attention layers.
	#	But decoder-only transformer model uses the "masked" multi-head self-attention layers.

	#--------------------
	# Positional encoding.
	detr_positional_encoding_test()  # Not yet implemented.

	#--------------------
	harvard_nlp_transformer_test()  # Not yet implemented.

	# PyTorch.
	#	REF [file] >> ../machine_learning/pytorch/pytorch_transformer.py
	# PyTorch Lightning.
	#	REF [file] >> ../machine_learning/pytorch_lightning/pl_transformer.py

	#--------------------
	# Hugging Face Transformers.
	#	REF [file] >> ./hugging_face_transformers_test.py

	#--------------------
	# GPT.
	#	REF [file] >> ./gpt_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
