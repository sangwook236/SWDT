#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/DemisEom/SpecAugment

import librosa

def tensorflow_example():
	import SpecAugment.spec_augment_tensorflow

	y, sr = librosa.load(librosa.example('nutcracker'), sr=None, mono=True)
	#y, sr = librosa.load(librosa.example('trumpet'), sr=None, mono=True)
	#y, sr = librosa.load('./stereo.ogg', sr=None, mono=True)

	mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=128, fmax=8000)
	print('Mel spectrogram: shape = {}, dtype = {}.'.format(mel_spectrogram.shape, mel_spectrogram.dtype))

	warped_masked_spectrogram = SpecAugment.spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)
	print('Mel spectrogram (augmented): shape = {}, dtype = {}.'.format(warped_masked_spectrogram.shape, warped_masked_spectrogram.dtype))
	#print(warped_masked_spectrogram)

	SpecAugment.spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram, 'Before augmentation')
	SpecAugment.spec_augment_tensorflow.visualization_spectrogram(warped_masked_spectrogram, 'After augmentation')

def pytorch_example():
	import torch
	import SpecAugment.spec_augment_pytorch

	y, sr = librosa.load(librosa.example('nutcracker'), sr=None, mono=True)
	#y, sr = librosa.load(librosa.example('trumpet'), sr=None, mono=True)
	#y, sr = librosa.load('./stereo.ogg', sr=None, mono=True)

	mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=128, fmax=8000)

	mel_spectrogram = torch.tensor(mel_spectrogram)
	mel_spectrogram = torch.unsqueeze(mel_spectrogram, dim=0)
	print('Mel spectrogram: shape = {}, dtype = {}.'.format(mel_spectrogram.shape, mel_spectrogram.dtype))

	warped_masked_spectrogram = SpecAugment.spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram, time_warping_para=50, frequency_masking_para=50, time_masking_para=1000, frequency_mask_num=2, time_mask_num=2)
	print('Mel spectrogram (augmented): shape = {}, dtype = {}.'.format(warped_masked_spectrogram.shape, warped_masked_spectrogram.dtype))
	#print(warped_masked_spectrogram)

	SpecAugment.spec_augment_pytorch.visualization_spectrogram(mel_spectrogram, 'Before augmentation')
	SpecAugment.spec_augment_pytorch.visualization_spectrogram(warped_masked_spectrogram, 'After augmentation')

def main():
	#tensorflow_example()
	pytorch_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
