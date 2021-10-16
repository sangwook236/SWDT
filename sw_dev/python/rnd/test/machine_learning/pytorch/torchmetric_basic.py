#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torchmetrics

# REF [site] >> https://devblog.pytorchlightning.ai/torchmetrics-v0-5-nlp-metrics-f4232467b0c5
def nlp_metrics_example():
	#import transformers
	#transformers.RobertaModel.from_pretrained("roberta-large")

	# BERTScore.
	#	BertScore works by encoding both a candidate and reference sentence using BERT and then calculate the cosine distance between their embeddings.
	#	This metric correlates well with human judgment on sentence-level.
	predictions = ["hello there", "general kenobi"]
	references = ["hello there", "master kenobi"]
	bertscore = torchmetrics.BERTScore("roberta-large")
	print("BERTScore(predictions, references) = {}.".format(bertscore(predictions, references)))

	# Bi-Lingual Evaluation Understudy (BLEU).
	#	A metric used for evaluating the quality of a machine-translated text, where quality refers to the correspondence between the machine and a human translation.
	#	BLEU was the first metric to claim a high correlation with a human judgement of quality.
	translate_corpus = ["the cat is on the mat".split()]
	reference_corpus = [["there is a cat on the mat".split(), "a cat is on the mat".split()]]
	bleu = torchmetrics.BLEUScore()
	print("BLEUScore(reference_corpus, translate_corpus) = {}.".format(bleu(reference_corpus, translate_corpus)))

	# Recall-Oriented Understudy for Gisting Evaluation (ROUGE).
	#	A metric that is also used for machine translation but also automatic summarization.
	#	ROUGE1: Overlap of unigrams (each word) between machine and reference.
	#	ROUGE2: Overlap of bigrams between machine and reference.
	#	ROUGE-L: Longest common sentence based statistics.
	targets = "Is your name John".split()
	preds = "My name is John".split()
	rouge = torchmetrics.ROUGEScore()
	print("ROUGEScore(preds, targets) = {}.".format(rouge(preds, targets)))

	# Word Error Rate (WER).
	#	WER works by first aligning the machine translation with the reference sentence and then calculate how many substitutions, deletions, insertions are needed to transform the machine translation to the reference sentence.
	predictions = ["this is the prediction", "there is an other sample"]
	references = ["this is the reference", "there is another sample"]
	wer = torchmetrics.WER("roberta-large")
	print("WER(predictions, references) = {}.".format(wer(predictions, references)))

	# Calibration error metrics.
	#	Calibration error metrics are important to measure how calibrated/overconfident your neural network is!
	import torch
	preds = torch.randn(100, 10).softmax(dim=-1)
	targets = torch.randint(10, (100,))
	print("torchmetrics.functional.calibration_error(preds, targets) = {}.".format(torchmetrics.functional.calibration_error(preds, targets)))

def main():
	nlp_metrics_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
