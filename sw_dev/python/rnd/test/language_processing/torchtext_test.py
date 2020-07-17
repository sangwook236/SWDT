#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://torchtext.readthedocs.io/en/latest/index.html
#	https://github.com/pytorch/text

import io, time
import torch, torchtext

# REF [site] >>
#	https://github.com/pytorch/text
#	https://torchtext.readthedocs.io/en/latest/examples.html
def tutorial_example():
	pos = torchtext.data.TabularDataset(
		path='./torchtext_data/pos/pos_wsj_train.tsv', format='tsv',
		fields=[('text', torchtext.data.Field()), ('labels', torchtext.data.Field())]
	)

	sentiment = torchtext.data.TabularDataset(
		path='./torchtext_data/sentiment/train.json', format='json',
		fields={
			'sentence_tokenized': ('text', torchtext.data.Field(sequential=True)),
			'sentiment_gold': ('labels', torchtext.data.Field(sequential=False))
		}
	)

	#--------------------
	my_custom_tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

	src = torchtext.data.Field(tokenize=my_custom_tokenizer)
	trg = torchtext.data.Field(tokenize=my_custom_tokenizer)
	mt_train = torchtext.datasets.TranslationDataset(
		path='./torchtext_data/mt/wmt16-ende.train', exts=('.en', '.de'),
		fields=(src, trg)
	)

	mt_dev = torchtext.data.TranslationDataset(
		path='./torchtext_data/mt/newstest2014', exts=('.en', '.de'),
		fields=(src, trg)
	)
	src.build_vocab(mt_train, max_size=80000)
	trg.build_vocab(mt_train, max_size=40000)
	# mt_dev shares the fields, so it shares their vocab objects.

	train_iter = torchtext.data.BucketIterator(
		dataset=mt_train, batch_size=32,
		sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg))
	)

	next(iter(train_iter))

	#--------------------
	TEXT = torchtext.data.Field()
	LABELS = torchtext.data.Field()

	train, val, test = torchtext.data.TabularDataset.splits(
		path='./torchtext_data/pos_wsj/pos_wsj', train='_train.tsv',
		validation='_dev.tsv', test='_test.tsv', format='tsv',
		fields=[('text', TEXT), ('labels', LABELS)]
	)

	train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
		(train, val, test), batch_sizes=(16, 256, 256),
		sort_key=lambda x: len(x.text), device=0
	)

	TEXT.build_vocab(train)
	LABELS.build_vocab(train)

def tokenizer_example():
	tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

	tokens = tokenizer('You can now install TorchText using pip!')
	print("tokenizer('You can now install TorchText using pip!') =", tokens)

def csv_iterator(data_filepath, ngrams):
	tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
	with io.open(data_filepath, encoding='utf8') as fd:
		reader = torchtext.utils.unicode_csv_reader(fd)
		for row in reader:
			tokens = ' '.join(row[1:])
			yield torchtext.data.utils.ngrams_iterator(tokenizer(tokens), ngrams)

# REF [site] >> https://github.com/pytorch/text/blob/master/examples/vocab/vocab.py
def vocab_example():
	csv_filepath = './torchtext_data/test.csv'
	vocab_filepath = './torchtext_data/vocab.pth'
	ngrams = 2

	vocab = torchtext.vocab.build_vocab_from_iterator(csv_iterator(csv_filepath, ngrams))

	torch.save(vocab, vocab_filepath)
	print('Saved a vocab to {}.'.format(vocab_filepath))

def main():
	#tutorial_example()  # No data.

	#tokenizer_example()
	vocab_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
