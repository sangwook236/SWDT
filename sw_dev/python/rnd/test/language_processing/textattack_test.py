#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/QData/TextAttack

import textattack

# REF [site] >> https://textattack.readthedocs.io/en/latest/examples/1_Introduction_and_Transformations.html
def simple_attack_example():
	# FIXME [implement] >>

	attacker = textattack.shared.Attack(
		goal_function=textattack.goal_functions.text.MinimizeBleu,
		#goal_function=textattack.goal_functions.text.NonOverlappingOutput,
		constraints=[],
		transformation=textattack.transformations.WordSwapEmbedding,
		search_method=textattack.search_methods.BeamSearch,
		transformation_cache_size=2**15,
        constraint_cache_size=2**15
	)

	#attackedText = textattack.shared.AttackedText(...)

def simple_augmentation_example():
	#augmenter = textattack.augmentation.EasyDataAugmenter(
	#augmenter = textattack.augmentation.SwapAugmenter(
	#augmenter = textattack.augmentation.SynonymInsertionAugmenter(
	#augmenter = textattack.augmentation.WordNetAugmenter(
	#augmenter = textattack.augmentation.DeletionAugmenter(
	augmenter = textattack.augmentation.EmbeddingAugmenter(
	#augmenter = textattack.augmentation.CharSwapAugmenter(
		pct_words_to_swap=0.1,  # Percentage of words to swap per augmented example. [0, 1].
		transformations_per_example=2  # Maximum number of augmentations.
	)

	txt = 'I would like to think while walking.'
	augmented = augmenter.augment(txt)
	print('{} --> {}'.format(txt, augmented))

def main():
	#simple_attack_example()  # Not yet implemented.
	simple_augmentation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
