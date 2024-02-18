import torch
import harvard_nlp_transformer

def batch_test():
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print(batch.src)  # [batch_size, seq_len]
	print(batch.tgt)  # [batch_size, seq_len - 1]
	print(batch.tgt_y)  # [batch_size, seq_len - 1]
	print(batch.src_mask)  # [batch_size, 1, seq_len]
	print(batch.tgt_mask)  # [batch_size, seq_len - 1, seq_len - 1]
	print(batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)
	print(batch.src.dtype, batch.tgt.dtype, batch.tgt_y.dtype, batch.src_mask.dtype, batch.tgt_mask.dtype)

def transformer_test():
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print(batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs = torch.max(srcs).item() + 1
	max_tgt_vocabs = torch.max(tgts).item() + 1
	model = harvard_nlp_transformer.make_model(max_src_vocabs, max_tgt_vocabs, N=2)

	# Check attention mechanism
	#	EncoderDecoder.forward()
	#		Encoder.forward()
	#			EncoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()
	#		Decoder.forward()
	#			DecoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()

	#-----
	model_outputs = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, max_tgt_vocabs]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print(model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_with_independent_embedding_test():
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print(batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs = torch.max(srcs).item() + 1
	max_tgt_vocabs = torch.max(tgts).item() + 1
	model, src_emb, tgt_emb = harvard_nlp_transformer.make_model_without_embedding(max_src_vocabs, max_tgt_vocabs, N=2)

	# Check attention mechanism
	#	EncoderDecoder.forward()
	#		Encoder.forward()
	#			EncoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()
	#		Decoder.forward()
	#			DecoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()

	#-----
	src_batch = src_emb(batch.src)  # [batch_size, seq_len, d_model]
	tgt_batch = tgt_emb(batch.tgt)  # [batch_size, seq_len - 1, d_model]

	print(src_batch.shape, tgt_batch.shape)

	#-----
	model_outputs = model(src_batch, tgt_batch, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, max_tgt_vocabs]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print(model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_with_token_span_embedding_test():
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print(batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	# Embeddings with more than 1 token span
	emb_token_span = 3
	batch.src_mask = batch.src_mask.repeat_interleave(emb_token_span, dim=-1)
	batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=1)
	batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=-1)

	#-----
	max_src_vocabs = torch.max(srcs).item() + 1
	max_tgt_vocabs = torch.max(tgts).item() + 1
	model, src_emb, tgt_emb = harvard_nlp_transformer.make_model_without_embedding(max_src_vocabs, max_tgt_vocabs, N=2)

	# Check attention mechanism
	#	EncoderDecoder.forward()
	#		Encoder.forward()
	#			EncoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()
	#		Decoder.forward()
	#			DecoderLayer.forward()
	#				MultiHeadedAttention.forward()
	#					attention()

	#-----
	src_batch = src_emb(batch.src)  # [batch_size, seq_len, d_model]
	tgt_batch = tgt_emb(batch.tgt)  # [batch_size, seq_len - 1, d_model]

	# Embeddings with more than 1 token span
	src_batch = src_batch.repeat_interleave(emb_token_span, dim=1)  # [batch_size, seq_len * emb_token_span, d_model]
	tgt_batch = tgt_batch.repeat_interleave(emb_token_span, dim=1)  # [batch_size, seq_len * emb_token_span, d_model]

	print(src_batch.shape, tgt_batch.shape)

	#-----
	model_outputs = model(src_batch, tgt_batch, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, max_tgt_vocabs]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print(model_outputs.shape, generator_outputs.shape, pred.shape)

def train_test():
	import time

	class DummyOptimizer(torch.optim.Optimizer):
		def __init__(self):
			self.param_groups = [{"lr": 0}]
			None

		def step(self):
			None

		def zero_grad(self, set_to_none=False):
			None

	class DummyScheduler:
		def step(self):
			None

	def data_gen(V, batch_size, nbatches):
		"Generate random data for a src-tgt copy task."
		for i in range(nbatches):
			data = torch.randint(1, V, size=(batch_size, 10))
			data[:, 0] = 1
			src = data.requires_grad_(False).clone().detach()
			tgt = data.requires_grad_(False).clone().detach()
			yield harvard_nlp_transformer.Batch(src, tgt, 0)

	class SimpleLossCompute:
		"A simple loss compute and train function."

		def __init__(self, generator, criterion):
			self.generator = generator
			self.criterion = criterion

		def __call__(self, x, y, norm):
			x = self.generator(x)
			sloss = (
				self.criterion(
					x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
				)
				/ norm
			)
			return sloss.data * norm, sloss

	class TrainState:
		"""Track number of steps, examples, and tokens processed"""

		step: int = 0  # Steps in the current epoch
		accum_step: int = 0  # Number of gradient accumulation steps
		samples: int = 0  # total # of examples used
		tokens: int = 0  # total # of tokens processed

	def run_epoch(
		data_iter,
		model,
		loss_compute,
		optimizer,
		scheduler,
		mode="train",
		accum_iter=1,
		train_state=TrainState(),
	):
		"""Train a single epoch"""
		start = time.time()
		total_tokens = 0
		total_loss = 0
		tokens = 0
		n_accum = 0
		for i, batch in enumerate(data_iter):
			out = model.forward(
				batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
			)
			loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
			# loss_node = loss_node / accum_iter
			if mode == "train" or mode == "train+log":
				loss_node.backward()
				train_state.step += 1
				train_state.samples += batch.src.shape[0]
				train_state.tokens += batch.ntokens
				if i % accum_iter == 0:
					optimizer.step()
					optimizer.zero_grad(set_to_none=True)
					n_accum += 1
					train_state.accum_step += 1
				scheduler.step()

			total_loss += loss
			total_tokens += batch.ntokens
			tokens += batch.ntokens
			if i % 40 == 1 and (mode == "train" or mode == "train+log"):
				lr = optimizer.param_groups[0]["lr"]
				elapsed = time.time() - start
				print(
					(
						"Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
						+ "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
					)
					% (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
				)
				start = time.time()
				tokens = 0
			del loss
			del loss_node
		return total_loss / total_tokens, train_state

	def rate(step, model_size, factor, warmup):
		"""
		we have to default the step to 1 for LambdaLR function
		to avoid zero raising to negative power.
		"""
		if step == 0:
			step = 1
		return factor * (
			model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
		)

	def example_simple_model():
		V = 11
		criterion = harvard_nlp_transformer.LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
		model = harvard_nlp_transformer.make_model(V, V, N=2)

		optimizer = torch.optim.Adam(
			model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
		)
		lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
			optimizer=optimizer,
			lr_lambda=lambda step: rate(
				step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
			),
		)

		batch_size = 80
		for epoch in range(20):
			model.train()
			run_epoch(
				data_gen(V, batch_size, 20),
				model,
				SimpleLossCompute(model.generator, criterion),
				optimizer,
				lr_scheduler,
				mode="train",
			)
			model.eval()
			run_epoch(
				data_gen(V, batch_size, 5),
				model,
				SimpleLossCompute(model.generator, criterion),
				DummyOptimizer(),
				DummyScheduler(),
				mode="eval",
			)[0]

		model.eval()
		src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
		max_len = src.shape[1]
		src_mask = torch.ones(1, 1, max_len)
		print(harvard_nlp_transformer.greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

	example_simple_model()

def main():
	# Harvard NLP transformer:
	#	https://nlp.seas.harvard.edu/annotated-transformer/
	#	https://github.com/harvardnlp/annotated-transformer

	batch_test()

	print("------------------------------------------------------------")
	transformer_test()
	print("------------------------------------------------------------")
	# When not performing embeddings in a transformer (when using external embeddings)
	transformer_with_independent_embedding_test()
	print("------------------------------------------------------------")
	# When having multiple token spans at a timestep like Decision Transformer
	transformer_with_token_span_embedding_test()

	print("------------------------------------------------------------")
	train_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
