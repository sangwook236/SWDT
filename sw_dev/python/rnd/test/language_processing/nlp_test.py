#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/mistralai
def voxtral_example():
	# Models:
	#	mistralai/Voxtral-Mini-3B-2507
	#	mistralai/Voxtral-Small-24B-2507

	# Install:
	#	pip install --upgrade mistral_common\[audio\]

	import torch
	import transformers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if True:
		# Multi-audio + text instruction

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		conversation = [
			{
				"role": "user",
				"content": [
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
					},
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
					},
					{"type": "text", "text": "What sport and what nursery rhyme are referenced?"},
				],
			}
		]

		inputs = processor.apply_chat_template(conversation)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated response:")
		print("=" * 80)
		print(decoded_outputs[0])
		print("=" * 80)

	if True:
		# Multi-turn

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		conversation = [
			{
				"role": "user",
				"content": [
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
					},
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
					},
					{"type": "text", "text": "Describe briefly what you can hear."},
				],
			},
			{
				"role": "assistant",
				"content": "The audio begins with the speaker delivering a farewell address in Chicago, reflecting on his eight years as president and expressing gratitude to the American people. The audio then transitions to a weather report, stating that it was 35 degrees in Barcelona the previous day, but the temperature would drop to minus 20 degrees the following day.",
			},
			{
				"role": "user",
				"content": [
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
					},
					{"type": "text", "text": "Ok, now compare this new audio with the previous one."},
				],
			},
		]

		inputs = processor.apply_chat_template(conversation)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated response:")
		print("=" * 80)
		print(decoded_outputs[0])
		print("=" * 80)

	if True:
		# Text only

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		conversation = [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Why should AI models be open-sourced?",
					},
				],
			}
		]

		inputs = processor.apply_chat_template(conversation)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated response:")
		print("=" * 80)
		print(decoded_outputs[0])
		print("=" * 80)

	if True:
		# Audio only

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		conversation = [
			{
				"role": "user",
				"content": [
					{
						"type": "audio",
						"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
					},
				],
			}
		]

		inputs = processor.apply_chat_template(conversation)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated response:")
		print("=" * 80)
		print(decoded_outputs[0])
		print("=" * 80)

	if True:
		# Batched inference

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		conversations = [
			[
				{
					"role": "user",
					"content": [
						{
							"type": "audio",
							"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
						},
						{
							"type": "audio",
							"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
						},
						{
							"type": "text",
							"text": "Who's speaking in the speach and what city's weather is being discussed?",
						},
					],
				}
			],
			[
				{
					"role": "user",
					"content": [
						{
							"type": "audio",
							"path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
						},
						{"type": "text", "text": "What can you tell me about this audio?"},
					],
				}
			],
		]

		inputs = processor.apply_chat_template(conversations)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated responses:")
		print("=" * 80)
		for decoded_output in decoded_outputs:
			print(decoded_output)
			print("=" * 80)

	if True:
		# Transcribe

		repo_id = "mistralai/Voxtral-Mini-3B-2507"
		#repo_id = "mistralai/Voxtral-Small-24B-2507"

		processor = transformers.AutoProcessor.from_pretrained(repo_id)
		model = transformers.VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

		inputs = processor.apply_transcrition_request(language="en", audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", model_id=repo_id)
		inputs = inputs.to(device, dtype=torch.bfloat16)

		outputs = model.generate(**inputs, max_new_tokens=500)
		decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

		print("\nGenerated responses:")
		print("=" * 80)
		for decoded_output in decoded_outputs:
			print(decoded_output)
			print("=" * 80)

def main():
	# Chat
	voxtral_example()  # Voxtral
	# Refer to canary_qwen_example() in ./speech_processing_test.py

#--------------------------------------------------------------------

if "_main__" == __name__:
	main()
