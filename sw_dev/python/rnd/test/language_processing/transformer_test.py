#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
def detr_positional_encoding_test():
	raise NotImplementedError

	#pe = PositionEmbeddingSine(num_pos_feats=64, temperature=10000, normalize=False, scale=None)
	#pe = PositionEmbeddingLearned(num_pos_feats=256)

# REF [site] >> https://github.com/microsoft/torchscale
def torchscale_test():
	# Install.
	#	pip install torchscale

	from torchscale.architecture.config import EncoderConfig, DecoderConfig, EncoderDecoderConfig
	from torchscale.architecture.encoder import Encoder
	from torchscale.architecture.decoder import Decoder
	from torchscale.architecture.encoder_decoder import EncoderDecoder

	# Creating an encoder model.
	config = EncoderConfig(vocab_size=64000)
	#config = EncoderConfig(vocab_size=64000, deepnorm=True, subln=True, use_xmoe=Tru, multiway=True, xpos_rel_pos=True)
	encoder = Encoder(config)

	print("Encoder:")
	print(encoder)

	# Creating a decoder model.
	config = DecoderConfig(vocab_size=64000)
	decoder = Decoder(config)

	print("Decoder:")
	print(decoder)

	# Creating an encoder-decoder model.
	config = EncoderDecoderConfig(vocab_size=64000)
	encdec = EncoderDecoder(config)

	print("EncoderDecoder:")
	print(encdec)

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
	# PyTorch.
	#	REF [file] >> ../machine_learning/pytorch/pytorch_transformer.py
	# PyTorch Lightning.
	#	REF [file] >> ../machine_learning/pytorch_lightning/pl_transformer.py

	# Hugging Face Transformers.
	#	REF [file] >> ./hugging_face_transformers_test.py

	# Harvard NLP transformer.
	#	REF [file] >> ./harvard_nlp_transformer_test.py

	torchscale_test()

	#--------------------
	# GPT.
	#	REF [file] >> ./gpt_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
