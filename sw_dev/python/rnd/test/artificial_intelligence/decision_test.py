#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Contrastive-LM/CLM
def clm_example():
	from clm import CLMClient, Choice, Noul, Score

	client = CLMClient()   # CLM_BASE_URL (default http://127.0.0.1:8700), CLM_API_KEY
	r = client.system_one(
		state="Customer: my invoice was charged twice and nobody answers the phone!",
		questions={
			"urgency": Noul(instructions="Is this urgent?"),
			"department": Choice(instructions="Which team should handle this?",
								criteria={"billing": "Charges, invoices, refunds",
										"technical": "Bugs and outages"}),
			"frustration": Score(instructions="How frustrated is the customer?",
								criteria=["Calm", "Frustrated", "Very angry"]),
		},
	)
	print(r.answers["urgency"].noul)                # 0.41022     probability the statement is true
	print(r.answers["department"].choice)           # billing
	print(r.answers["department"].probabilities)    # {'billing': 0.93878, 'technical': 0.06122}
	print(r.answers["frustration"].score)           # 1.98386     expected level, 0..2
	print(r.usage.input_tokens, r.latency_ms)       # 38 58.1     (106 tokens on a cold cache: option texts are embedded once)

# REF [site] >> https://aistudio.google.com/docs/models/gemini-decision-1
def gemini_decision_1_example():
	from google import genai

	client = genai.Client(http_options={"api_version": "v1beta"})

	result = client.decisions.create(
		model="gemini-decision-1",
		input="The blue mug I ordered on Monday arrived cracked on Wednesday.",
		questions={
			"is_damaged": {
				"type": "noul",
				"instructions": "Did the item arrive damaged?",
			},
			"queue": {
				"type": "choice",
				"instructions": "Which support queue should handle this message?",
				"criteria": {
					"billing": "Payment, invoice, or refund questions.",
					"replacements": "Damaged, missing, or wrong items.",
					"general": "Product questions and account updates.",
				},
			},
		},
	)

	damaged = result.answers["is_damaged"]
	queue = result.answers["queue"]
	print(f"P(damaged)={damaged.noul:.2f} queue={queue.choice}")

# REF [site] >> https://huggingface.co/kakaocorp
def kanana_decide_example():
	# Models:
	#	kakaocorp/kanana-2-v-decide-3b-2611

	if True:
		import torch
		import transformers

		model = transformers.AutoModel.from_pretrained("kakaocorp/kanana-2-v-decide-3b-2611", trust_remote_code=True,
													dtype=torch.bfloat16).to("cuda")

		if True:
			result = model.system_one(
				state="The item I ordered arrived damaged. Please replace it.",
				questions={
					"category": {"type": "choice", "instructions": "Classify the customer's inquiry.",
								"criteria": {"delivery": "Delivery inquiry", "exchange": "Replacement request", "payment": "Payment inquiry"}},
					"damaged": {"type": "noul", "instructions": "Is the item damaged?"},
					"sentiment": {"type": "score", "instructions": "How positive is the customer's sentiment?",
								"criteria": ["Negative", "Neutral", "Positive"]},
				},
			)
		else:
			result = model.system_one(state="The customer attached a photo of the delivered item.",
									images=["photo.jpg"],
									questions={"damaged": {"type": "noul", "instructions": "Is the item damaged?"}})
		print(result["answers"]["category"]["choice"])   # "exchange"
		print(result["answers"]["damaged"]["noul"])      # probability of yes (0-1)
		print(result["answers"]["sentiment"]["score"])   # expected level (0-2)

	if True:
		import transformers

		decide = transformers.pipeline("decision", model="kakaocorp/kanana-2-v-decide-3b-2611", trust_remote_code=True, device=0)
		decide(state="The parcel never arrived.", questions={"refund": {"type": "noul", "instructions": "Does the customer want a refund?"}})

def main():
	#clm_example()  # Contrastive Language Model (CLM)
	#gemini_decision_1_example()  # Gemini Decision 1
	kanana_decide_example()  # Kanana Decision Model

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
