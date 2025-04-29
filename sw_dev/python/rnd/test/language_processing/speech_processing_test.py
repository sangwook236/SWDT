#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import transformers

# REF [site] >> https://huggingface.co/microsoft
def microsoft_asr_example():
	import datasets

	dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
	dataset = dataset.sort("id")
	sampling_rate = dataset.features["audio"].sampling_rate
	example_speech = dataset[0]["audio"]["array"]

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
	model = transformers.SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

	inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

	predicted_ids = model.generate(**inputs, max_length=100)

	transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
	print(transcription[0])

# REF [site] >> https://huggingface.co/nvidia
def nvidia_asr_example():
	# Models:
	#	en, es, uk, fr, it, de, pl, hr, be, ca, ua, ru, zh.
	#
	#	nvidia/stt_en_conformer_ctc_small.
	#	nvidia/stt_en_conformer_ctc_large.
	#	nvidia/stt_en_conformer_transducer_large.
	#	nvidia/stt_en_conformer_transducer_xlarge.
	#	nvidia/stt_en_fastconformer_ctc_large.
	#	nvidia/stt_en_fastconformer_ctc_xlarge.
	#	nvidia/stt_en_fastconformer_transducer_large.
	#	nvidia/stt_en_fastconformer_transducer_xlarge.
	#	nvidia/stt_en_fastconformer_transducer_xxlarge.
	#	nvidia/stt_en_fastconformer_hybrid_large_pc.
	#	nvidia/stt_en_citrinet_256_ls.
	#	nvidia/stt_en_citrinet_384_ls.
	#	nvidia/stt_en_citrinet_512_ls.
	#	nvidia/stt_en_citrinet_768_ls.
	#	nvidia/stt_en_citrinet_1024_ls.
	#	nvidia/stt_en_citrinet_1024_gamma_0_25.
	#
	#	nvidia/parakeet-rnnt-0.6b.
	#	nvidia/parakeet-rnnt-1.1b.
	#	nvidia/parakeet-ctc-0.6b.
	#	nvidia/parakeet-ctc-1.1b.
	#	nvidia/parakeet-tdt-1.1b.
	#
	#	nvidia/canary-1b.

	if False:
		# Install:
		#	pip install nemo_toolkit['all']

		#model_name = "nvidia/stt_en_fastconformer_transducer_large"
		model_name = "nvidia/parakeet-rnnt-1.1b"
		#model_name = "nvidia/parakeet-tdt-1.1b"

		import nemo.collections.asr as nemo_asr
		asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)

		# Get a sample file.
		#	wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
		transcribed = asr_model.transcribe(["./2086-149220-0033.wav"])
		print(transcribed)

	if False:
		# Install:
		#	pip install nemo_toolkit['all']

		model_name = "nvidia/parakeet-ctc-1.1b"

		import nemo.collections.asr as nemo_asr
		asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)

		# Get a sample file.
		#	wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
		transcribed = asr_model.transcribe(["./2086-149220-0033.wav"])
		print(transcribed)

	if True:
		# Install:
		#	pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]

		from nemo.collections.asr.models import EncDecMultiTaskModel

		# Load model
		canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")

		# Update dcode params
		decode_cfg = canary_model.cfg.decoding
		decode_cfg.beam.beam_size = 1
		canary_model.change_decoding_strategy(decode_cfg)

		if True:
			predicted_text = canary_model.transcribe(
				paths2audio_files=["path1.wav", "path2.wav"],
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

		if True:
			# Automatic speech-to-text recognition (ASR)
			'''
			# Example of a line in input_manifest.json
			{
				"audio_filepath": "/path/to/audio.wav",  # Path to the audio file
				"duration": 1000,  # Duration of the audio, can be set to `None` if using NeMo main branch
				"taskname": "asr",  # Use "s2t_translation" for speech-to-text translation with r1.23, or "ast" if using the NeMo main branch
				"source_lang": "en",  # Language of the audio input, set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
				"target_lang": "en",  # Language of the text output, choices=['en','de','es','fr']
				"pnc": "yes",  # Whether to have PnC output, choices=['yes', 'no']
				"answer": "na", 
			}
			'''

			predicted_text = canary_model.transcribe(
				"/path/to/input_manifest.json",
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

		if True:
			# Automatic speech-to-text translation (AST)
			'''
			# Example of a line in input_manifest.json
			{
				"audio_filepath": "/path/to/audio.wav",  # Path to the audio file
				"duration": 1000,  # Duration of the audio, can be set to `None` if using NeMo main branch
				"taskname": "s2t_translation", # r1.23 only recognizes "s2t_translation", but "ast" is supported if using the NeMo main branch
				"source_lang": "en", # Language of the audio input, choices=['en','de','es','fr']
				"target_lang": "de", # Language of the text output, choices=['en','de','es','fr']
				"pnc": "yes",  # Whether to have PnC output, choices=['yes', 'no']
				"answer": "na" 
			}
			'''

			predicted_text = canary_model.transcribe(
				"/path/to/input_manifest.json",
				batch_size=16,  # Batch size to run the inference with
			)
			print(predicted_text)

def openai_asr_example():
	# Models:
	#	openai/whisper-tiny.
	#	openai/whisper-base.
	#	openai/whisper-small.
	#	openai/whisper-medium.
	#	openai/whisper-large: ~6.17GB.
	#	openai/whisper-large-v2.

	import datasets

	pretrained_model_name = "openai/whisper-large"

	if True:
		# Transcription (English to English).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		model.config.forced_decoder_ids = None

		# Load dummy dataset and read audio files.
		ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
		sample = ds[0]["audio"]
		input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

		# Generate token ids.
		predicted_ids = model.generate(input_features)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
		print(f"Transcription (w/ special tokens): {transcription}.")
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription (w/o special tokens): {transcription}.")

	if False:
		# Transcription (French to French).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

		# Load streaming dataset and read first audio sample.
		ds = datasets.load_dataset("common_voice", "fr", split="test", streaming=True)
		ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
		input_speech = next(iter(ds))["audio"]
		input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

		# Generate token ids.
		predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids)
		print(f"Transcription (w/ special tokens): {transcription}.")
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription (w/o special tokens): {transcription}.")

	if True:
		# Translation (French to English).

		# Load model and processor.
		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)
		forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

		# Load streaming dataset and read first audio sample.
		ds = datasets.load_dataset("common_voice", "fr", split="test", streaming=True)
		ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
		input_speech = next(iter(ds))["audio"]
		input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

		# Generate token ids.
		predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

		# Decode token ids to text.
		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		print(f"Transcription: {transcription}.")

	if False:
		# Evaluation.

		import evaluate

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Device: {device}.")

		processor = transformers.WhisperProcessor.from_pretrained(pretrained_model_name)
		model = transformers.WhisperForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)

		def map_to_pred(batch):
			audio = batch["audio"]
			input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
			batch["reference"] = processor.tokenizer._normalize(batch["text"])

			model.eval()
			with torch.no_grad():
				predicted_ids = model.generate(input_features.to(device))[0]
			transcription = processor.decode(predicted_ids)
			batch["prediction"] = processor.tokenizer._normalize(transcription)
			return batch

		librispeech_test_clean = datasets.load_dataset("librispeech_asr", "clean", split="test")
		result = librispeech_test_clean.map(map_to_pred)

		wer = evaluate.load("wer")
		print(f'WER = {100 * wer.compute(references=result["reference"], predictions=result["prediction"])}.')

	if False:
		# Long-form transcription.
		#	The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration.
		#	However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length.
		#	This is possible through Transformers pipeline method.
		#	Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline.
		#	It can also be extended to predict utterance level timestamps by passing return_timestamps=True.

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(f"Device: {device}.")

		pipe = transformers.pipeline(
			"automatic-speech-recognition",
			model=pretrained_model_name,
			chunk_length_s=30,
			device=device,
		)

		ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
		sample = ds[0]["audio"]

		prediction = pipe(sample.copy())["text"]
		print(f"Prediction: {prediction}.")

		# We can also return timestamps for the predictions.
		prediction = pipe(sample, return_timestamps=True)["chunks"]
		print(f"Prediction: {prediction}.")

# REF [site] >> https://huggingface.co/speechbrain
def speech_brain_asr_example():
	import speechbrain

	asr_model = speechbrain.pretrained.WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fr", savedir="pretrained_models/asr-whisper-large-v2-commonvoice-fr")
	#asr_model = speechbrain.pretrained.WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fr", savedir="pretrained_models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device": "cuda"})

	# https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fr/tree/main
	asr_model.transcribe_file("./example-fr.mp3")

# REF [site] >> https://huggingface.co/microsoft
def microsoft_tts_example():
	import datasets
	import soundfile as sf

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
	model = transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
	vocoder = transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

	inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

	# Load xvector containing speaker's voice characteristics from a dataset.
	embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
	speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

	speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

	sf.write("./speecht5_tts.wav", speech.numpy(), samplerate=16000)

# REF [site] >> https://huggingface.co/nvidia
def nvidia_tts_example():
	# Models:
	#	nvidia/tts_hifigan.
	#	nvidia/tts_en_fastpitch.

	# Install:
	#	pip install nemo_toolkit['all']

	import soundfile as sf

	# Load FastPitch.
	from nemo.collections.tts.models import FastPitchModel
	spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")

	# Load vocoder.
	from nemo.collections.tts.models import HifiGanModel
	model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

	# Generate audio.
	parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
	spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
	audio = model.convert_spectrogram_to_audio(spec=spectrogram)

	# Save the audio to disk in a file called speech.wav
	sf.write("./speech.wav", audio.to("cpu").numpy(), 22050)

# REF [site] >> https://huggingface.co/speechbrain
def speech_brain_tts_example():
	import speechbrain
	import torchaudio

	if True:
		# Intialize TTS (tacotron2) and Vocoder (HiFiGAN).
		tacotron2 = speechbrain.pretrained.Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
		hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
		#tacotron2 = speechbrain.pretrained.Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device": "cuda"})
		#hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts={"device": "cuda"})

		# Running the TTS.
		if True:
			input_text = "Mary had a little lamb"
		else:
			input_text = [
				"A quick brown fox jumped over the lazy dog",
				"How much wood would a woodchuck chuck?",
				"Never odd or even"
			]
		mel_output, mel_length, alignment = tacotron2.encode_batch(input_text)  # torch.Tensor.

		# Running Vocoder (spectrogram-to-waveform).
		waveforms = hifi_gan.decode_batch(mel_output)  # torch.Tensor.

		# Save the waverform.
		torchaudio.save('./speechbrain_tts_tacotron2.wav', waveforms.squeeze(dim=1).cpu(), 22050)

	if True:
		# Intialize TTS (fastspeech2) and Vocoder (HiFiGAN).
		fastspeech2 = speechbrain.pretrained.FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="tmpdir_tts")
		hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder")
		#fastspeech2 = speechbrain.pretrained.FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="tmpdir_tts", run_opts={"device": "cuda"})
		#hifi_gan = speechbrain.pretrained.HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir_vocoder", run_opts={"device": "cuda"})

		# Running the TTS.
		if True:
			input_text = "Mary had a little lamb"
		else:
			input_text = [
				"A quick brown fox jumped over the lazy dog",
				"How much wood would a woodchuck chuck?",
				"Never odd or even"
			]
		mel_output, durations, pitch, energy = fastspeech2.encode_text(input_text)  # torch.Tensor.

		# Running Vocoder (spectrogram-to-waveform).
		waveforms = hifi_gan.decode_batch(mel_output)  # torch.Tensor.

		# Save the waverform.
		torchaudio.save('./speechbrain_tts_fastspeech2.wav', waveforms.squeeze(dim=1).cpu(), 16000)

# REF [site] >> https://huggingface.co/tensorspeech
def tensor_speech_tts_example():
	# Models:
	#	tensorspeech/tts-tacotron2-ljspeech-en.
	#	tensorspeech/tts-tacotron2-kss-ko.
	#	tensorspeech/tts-tacotron2-baker-ch.
	#	tensorspeech/tts-tacotron2-thorsten-ger.
	#	tensorspeech/tts-tacotron2-synpaflex-fr.
	#
	#	tensorspeech/tts-fastspeech-ljspeech-en.
	#	tensorspeech/tts-fastspeech2-ljspeech-en.
	#	tensorspeech/tts-fastspeech2-kss-ko.
	#	tensorspeech/tts-fastspeech2-baker-ch.
	#
	#	tensorspeech/tts-mb_melgan-ljspeech-en.
	#	tensorspeech/tts-melgan-ljspeech-en.
	#	tensorspeech/tts-mb_melgan-kss-ko.
	#	tensorspeech/tts-mb_melgan-baker-ch.
	#	tensorspeech/tts-mb_melgan-thorsten-ger.
	#	tensorspeech/tts-mb_melgan-synpaflex-fr.

	# Install:
	#	pip install TensorFlowTTS

	import tensorflow as tf
	import tensorflow_tts
	import soundfile as sf

	if True:
		if True:
			# English.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
			tacotron2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
			mb_melgan = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

			text = "This is a demo to show how to use our model to generate mel spectrogram from raw text."
		else:
			# Korean.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
			tacotron2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
			mb_melgan = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")

			text = "신은 우리의 수학 문제에는 관심이 없다. 신은 다만 경험적으로 통합할 뿐이다."

		input_ids = processor.text_to_sequence(text)

		# Tacotron2 inference (text-to-mel).
		decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
			input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
			input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
			speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
		)

		#-----
		# MelGAN inference (mel-to-wav).
		audio = mb_melgan.inference(mel_outputs)[0, :, 0]

		# Save to file.
		sf.write('./tensorspeech_tts_tacotron2.wav', audio, 22050, "PCM_16")

	if True:
		if True:
			# English.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
			fastspeech2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

			text = "How are you?"
		else:
			# Korean.

			processor = tensorflow_tts.inference.AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
			fastspeech2 = tensorflow_tts.inference.TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")

			text = "신은 우리의 수학 문제에는 관심이 없다. 신은 다만 경험적으로 통합할 뿐이다."

		input_ids = processor.text_to_sequence(text)

		# FastSpeech2 inference.
		mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
			input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
			speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
			speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
			f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
			energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
		)

# REF [site] >> https://huggingface.co/microsoft
def microsoft_voice_conversion_example():
	import numpy as np
	import datasets
	import soundfile as sf

	dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
	dataset = dataset.sort("id")
	sampling_rate = dataset.features["audio"].sampling_rate
	example_speech = dataset[0]["audio"]["array"]

	processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
	model = transformers.SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
	vocoder = transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

	inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

	# Load xvector containing speaker's voice characteristics from a file.
	speaker_embeddings = np.load("./xvector_speaker_embedding.npy")
	speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

	speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

	sf.write("./speecht5_vc.wav", speech.numpy(), samplerate=16000)

def main():
	# Speech recognition.

	#microsoft_asr_example()  # SpeechT5.
	#nvidia_asr_example()  # Conformer, Citrinet, Parakeet, Canary.
	#openai_asr_example()  # Whisper.
	#speech_brain_asr_example()  # Whisper. Error.

	#-----
	# Speech synthesis.

	#microsoft_tts_example()  # SpeechT5.
	#nvidia_tts_example()  # FastPitch + HiFiGAN.
	#speech_brain_tts_example()  # Tacotron / FastSpeech + HiFiGAN.
	#tensor_speech_tts_example()  # Tacotron / FastSpeech + MelGAN.

	#-----
	# Speech-to-speech.

	#microsoft_voice_conversion_example()  # SpeechT5.

#--------------------------------------------------------------------

if "_main__" == __name__:
	main()
