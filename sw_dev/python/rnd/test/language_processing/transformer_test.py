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
	# Positional encoding.
	detr_positional_encoding_test()  # Not yet implemented.

	#--------------------
	harvard_nlp_transformer_test()  # Not yet implemented.

	#--------------------
	# Hugging Face transformers.
	#	REF [file] >> ./hugging_face_transformers_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
