#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/makcedward/nlpaug

import nlpaug
import nlpaug.augmenter, nlpaug.augmenter.audio, nlpaug.augmenter.spectrogram
import nlpaug.augmenter.char, nlpaug.augmenter.word, nlpaug.augmenter.sentence
import librosa
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb
def audio_example():
	wav_filepath = './Yamaha-V50-Rock-Beat-120bpm.wav'
	data, sr = librosa.load(wav_filepath)

	#--------------------
	# Crop augmenter.

	augmenter = nlpaug.augmenter.audio.CropAug(sampling_rate=sr)
	augmented_data = augmenter.augment(data)

	plt.figure('Crop Augmenter')
	librosa.display.waveplot(augmented_data, sr=sr, alpha=0.5)
	librosa.display.waveplot(data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Loudness augmenter.

	augmenter = nlpaug.augmenter.audio.LoudnessAug(loudness_factor=(2, 5))
	augmented_data = augmenter.augment(data)

	plt.figure('Loudness Augmenter')
	librosa.display.waveplot(augmented_data, sr=sr, alpha=0.5)
	librosa.display.waveplot(data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Mask augmenter.

	augmenter = nlpaug.augmenter.audio.MaskAug(sampling_rate=sr, mask_with_noise=False)
	augmented_data = augmenter.augment(data)

	plt.figure('Mask Augmenter')
	librosa.display.waveplot(data, sr=sr, alpha=0.5)
	librosa.display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Noise augmenter.

	augmenter = nlpaug.augmenter.audio.NoiseAug(noise_factor=0.03)
	augmented_data = augmenter.augment(data)

	plt.figure('Noise Augmenter')
	librosa.display.waveplot(data, sr=sr, alpha=0.5)
	librosa.display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Pitch augmenter.

	augmenter = nlpaug.augmenter.audio.PitchAug(sampling_rate=sr, pitch_factor=(2, 3))
	augmented_data = augmenter.augment(data)

	plt.figure('Pitch Augmenter')
	librosa.display.waveplot(data, sr=sr, alpha=0.5)
	librosa.display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Shift augmenter.

	augmenter = nlpaug.augmenter.audio.ShiftAug(sampling_rate=sr)
	augmented_data = augmenter.augment(data)

	plt.figure('Shift Augmenter')
	librosa.display.waveplot(data, sr=sr, alpha=0.5)
	librosa.display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# Speed augmenter.

	augmenter = nlpaug.augmenter.audio.SpeedAug()
	augmented_data = augmenter.augment(data)

	plt.figure('Speed Augmenter')
	librosa.display.waveplot(data, sr=sr, alpha=0.5)
	librosa.display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

	plt.tight_layout()
	plt.show()

	#--------------------
	# VTLP augmenter.

	augmenter = nlpaug.augmenter.audio.VtlpAug(sampling_rate=sr)
	augmented_data = augmenter.augment(data)

	nlpaug.util.visual.VisualWave.freq_power('VTLP Augmenter', data, sr, augmented_data)

	plt.tight_layout()
	plt.show()

# REF [site] >> https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
def textual_example():
	text = 'The quick brown fox jumps over the lazy dog .'

	#--------------------
	# (Character) OCR augmenter.

	# Substitute character by pre-defined OCR error.
	augmenter = nlpaug.augmenter.char.OcrAug()
	augmented_texts = augmenter.augment(text, n=3)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	for augmented in augmented_texts:
		print('\t{}'.format(augmented))

	#--------------------
	# (Character) Keyboard augmenter.

	# Substitute character by keyboard distance.
	augmenter = nlpaug.augmenter.char.KeyboardAug()
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Character) Random augmenter.

	# Insert character randomly.
	augmenter = nlpaug.augmenter.char.RandomCharAug(action='insert')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Substitute character randomly.
	augmenter = nlpaug.augmenter.char.RandomCharAug(action='substitute')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Swap character randomly.
	augmenter = nlpaug.augmenter.char.RandomCharAug(action='swap')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Delete character randomly.
	augmenter = nlpaug.augmenter.char.RandomCharAug(action='delete')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Spelling augmenter.

	# Substitute word by spelling mistake words dictionary.
	augmenter = nlpaug.augmenter.word.SpellingAug()
	augmented_text = augmenter.augment(text, n=3)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	for augmented in augmented_texts:
		print('\t{}'.format(augmented))

	#--------------------
	# (Word) Word embeddings augmenter.

	model_path = './GoogleNews-vectors-negative300.bin'

	# Insert word randomly by word embeddings similarity.
	augmenter = nlpaug.augmenter.word.WordEmbsAug(
		model_type='word2vec',  # {'word2vec', 'glove', 'fasttext'}.
		model_path=model_path,
		action='insert'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Substitute word by word2vec similarity.
	augmenter = nlpaug.augmenter.word.WordEmbsAug(
		model_type='word2vec',  # {'word2vec', 'glove', 'fasttext'}.
		model_path=model_path,
		action='substitute'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) TF-IDF augmenter.

	model_path = './model_dir'

	# Insert word by TF-IDF similarity.
	augmenter = nlpaug.augmenter.word.TfIdfAug(
		model_path=model_path,
		action='insert'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Substitute word by TF-IDF similarity.
	augmenter = nlpaug.augmenter.word.TfIdfAug(
		model_path=model_path,
		action='substitute'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Contextual word embeddings augmenter.

	# Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet).
	augmenter = nlpaug.augmenter.word.ContextualWordEmbsAug(
		model_path='bert-base-uncased',  # {'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'}.
		action='insert'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet).
	augmenter = nlpaug.augmenter.word.ContextualWordEmbsAug(
		model_path='bert-base-uncased',  # {'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'}.
		action='substitute'
	)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Synonym augmenter.

	# Substitute word by WordNet's synonym.
	augmenter = nlpaug.augmenter.word.SynonymAug(aug_src='wordnet')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Substitute word by PPDB's synonym.
	model_path = './ppdb-2.0-s-all'
	augmenter = nlpaug.augmenter.word.SynonymAug(aug_src='ppdb', model_path=model_path)
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Antonym augmenter.

	# Substitute word by antonym.
	augmenter = nlpaug.augmenter.word.AntonymAug()
	text1 = 'Good boy'
	augmented_text = augmenter.augment(text1)
	print('Original:')
	print('\t{}'.format(text1))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Random word augmenter.

	# Swap word randomly.
	augmenter = nlpaug.augmenter.word.RandomWordAug(action='swap')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Delete word randomly.
	augmenter = nlpaug.augmenter.word.RandomWordAug()
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	# Delete a set of contunous word will be removed randomly.
	augmenter = nlpaug.augmenter.word.RandomWordAug(action='crop')
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Split augmenter.

	# Split word to two tokens randomly.
	augmenter = nlpaug.augmenter.word.SplitAug()
	augmented_text = augmenter.augment(text)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Word) Back translation augmenter.

	# Swap word randomly.
	augmenter = nlpaug.augmenter.word.BackTranslationAug(
		from_model_name='transformer.wmt19.en-de',
		to_model_name='transformer.wmt19.de-en'
	)
	text1 = 'The quick brown fox jumped over the lazy dog'
	augmented_text = augmenter.augment(text1)
	print('Original:')
	print('\t{}'.format(text1))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

	#--------------------
	# (Sentence) Contextual word embeddings for sentence augmenter.

	# Insert sentence by contextual word embeddings (GPT2 or XLNet).
	augmenter = nlpaug.augmenter.sentence.ContextualWordEmbsForSentenceAug(
		model_path='xlnet-base-cased'  # {'gpt2', 'distilgpt2', 'xlnet-base-cased'}.
	)
	augmented_text = augmenter.augment(text, n=3)
	print('Original:')
	print('\t{}'.format(text))
	print('Augmented:')
	for augmented in augmented_texts:
		print('\t{}'.format(augmented))

	#--------------------
	# (Sentence) Abstractive summarization augmenter.

	article = """
The history of natural language processing (NLP) generally started in the 1950s, although work can be 
found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and 
Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. 
The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian 
sentences into English. The authors claimed that within three or five years, machine translation would
be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, 
which found that ten-year-long research had failed to fulfill the expectations, funding for machine 
translation was dramatically reduced. Little further research in machine translation was conducted 
until the late 1980s when the first statistical machine translation systems were developed.
"""

	augmenter = nlpaug.augmenter.sentence.AbstSummAug(model_path='t5-base', num_beam=3)
	augmented_text = augmenter.augment(article)
	print('Original:')
	print('\t{}'.format(article))
	print('Augmented:')
	print('\t{}'.format(augmented_text))

# REF [site] >> https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb
def spectrogram_example():
	wav_filepath = './Yamaha-V50-Rock-Beat-120bpm.wav'

	mel_spectrogram = nlpaug.util.audio.loader.AudioLoader.load_mel_spectrogram(wav_filepath, n_mels=128)
	nlpaug.util.audio.visualizer.AudioVisualizer.spectrogram('Original', mel_spectrogram)

	#--------------------
	# Frequency masking.

	augmenter = nlpaug.augmenter.spectrogram.FrequencyMaskingAug(mask_factor=80)

	augmented_mel_spectrogram = augmenter.substitute(mel_spectrogram)
	nlpaug.util.audio.visualizer.AudioVisualizer.spectrogram('Frequency Masking', augmented_mel_spectrogram)

	#--------------------
	# Time masking.

	augmenter = nlpaug.augmenter.spectrogram.TimeMaskingAug(mask_factor=80)

	augmented_mel_spectrogram = augmenter.substitute(mel_spectrogram)
	nlpaug.util.audio.visualizer.AudioVisualizer.spectrogram('Time Masking', augmented_mel_spectrogram)

	#--------------------
	# Combine frequency masking and time masking.

	flow = nlpaug.flow.Sequential([
		nlpaug.augmenter.spectrogram.FrequencyMaskingAug(mask_factor=50), 
		nlpaug.augmenter.spectrogram.TimeMaskingAug(mask_factor=20), 
		nlpaug.augmenter.spectrogram.TimeMaskingAug(mask_factor=30)
	])
	augmented_mel_spectrogram = flow.augment(mel_spectrogram)
	nlpaug.util.audio.visualizer.AudioVisualizer.spectrogram('Combine Frequency Masking and Time Masking', augmented_mel_spectrogram)

# REF [site] >> https://github.com/makcedward/nlpaug/blob/master/example/change_log.ipynb
def change_log_example():
	text = 'The quick brown fox jumps over the lazy dog.'

	# Sentence augmenter.
	aug = nlpaug.augmenter.sentence.ContextualWordEmbsForSentenceAug(model_path='gpt2', include_detail=True)
	augmented_data, change_logs = aug.augment(text)
	print('Augmented data: {}.'.format(augmented_data))
	for change_log in reversed(change_logs):
		print('Change log: {}.'.format(change_log))
		break

	# Word augmenter.
	aug = nlpaug.augmenter.word.ContextualWordEmbsAug(model_path='bert-base-uncased', include_detail=True)
	augmented_data, change_log = aug.augment(text)
	print('Augmented data: {}.'.format(augmented_data))
	print('Change log: {}.'.format(change_log))

	# Character augmenter.
	aug = nlpaug.augmenter.char.KeyboardAug(include_detail=True)
	augmented_data, change_log = aug.augment(text)
	print('Augmented data: {}.'.format(augmented_data))
	print('Change log: {}.'.format(change_log))

	# Pipeline.
	aug = nlpaug.flow.Sequential([
		nlpaug.augmenter.word.RandomWordAug(action='substitute', target_words=['A'], name='aug1', include_detail=False),
		nlpaug.flow.Sequential([
			nlpaug.augmenter.word.RandomWordAug(action='substitute', target_words=['D'],name='aug2', include_detail=False),
			nlpaug.augmenter.word.RandomWordAug(name='aug3', include_detail=True)
		], include_detail=False, name='pipe2')
	], include_detail=True, name='pipe1')
	augmented_data, change_log = aug.augment(text)
	print('Augmented data: {}.'.format(augmented_data))
	print('Change log: {}.'.format(change_log))

def main():
	#audio_example()
	textual_example()
	#spectrogram_example()
	#change_log_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
