import typing
import torch
import harvard_nlp_transformer

def batch_test():
	print("------------------------------------------------------------")

	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print("Batch:")
	print(batch.src)  # [batch_size, seq_len]
	print(batch.tgt)  # [batch_size, seq_len - 1]
	print(batch.tgt_y)  # [batch_size, seq_len - 1]
	print(batch.src_mask)  # [batch_size, 1, seq_len]
	print(batch.tgt_mask)  # [batch_size, seq_len - 1, seq_len - 1]
	print(batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)
	print(batch.src.dtype, batch.tgt.dtype, batch.tgt_y.dtype, batch.src_mask.dtype, batch.tgt_mask.dtype)

def transformer_test():
	print("------------------------------------------------------------")

	# Data
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print("Batch:", batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	# Model
	max_src_vocabs, max_tgt_vocabs = torch.max(srcs).item() + 1, torch.max(tgts).item() + 1
	n_enc_layers = n_dec_layers = 2
	model = harvard_nlp_transformer.make_model(max_src_vocabs, max_tgt_vocabs, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers)

	# NOTE [info] >> attention mechanism
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
	# When using IDs as input
	model_outputs = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, max_tgt_vocabs]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print("Model output:", model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_using_external_modules_test():
	print("------------------------------------------------------------")

	import copy

	# Data
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print("Batch:", batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs, max_tgt_vocabs = torch.max(srcs).item() + 1, torch.max(tgts).item() + 1
	d_model = 512
	n_head = 8
	d_ff = 2048
	n_enc_layers = n_dec_layers = 2
	dropout_prob = 0.1

	# Embeddings
	position = harvard_nlp_transformer.PositionalEncoding(d_model, dropout=dropout_prob)
	src_emb = torch.nn.Sequential(harvard_nlp_transformer.Embeddings(d_model, max_src_vocabs), copy.deepcopy(position))
	tgt_emb = torch.nn.Sequential(harvard_nlp_transformer.Embeddings(d_model, max_tgt_vocabs), copy.deepcopy(position))
	# Generator
	generator = harvard_nlp_transformer.Generator(d_model, max_tgt_vocabs)

	# Model
	model = harvard_nlp_transformer.make_model_using_external_modules(src_emb, tgt_emb, generator, d_model=d_model, n_head=n_head, d_ff=d_ff, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers, dropout=dropout_prob)

	#-----
	# When using IDs as input
	model_outputs = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, d_output]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print("Model output:", model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_without_embeddings_test():
	print("------------------------------------------------------------")

	import copy

	# Data
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

	print("Batch:", batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs, max_tgt_vocabs = torch.max(srcs).item() + 1, torch.max(tgts).item() + 1
	d_model = 512
	n_head = 8
	d_ff = 2048
	n_enc_layers = n_dec_layers = 2
	dropout_prob = 0.1

	# Generator
	generator = harvard_nlp_transformer.Generator(d_model, max_tgt_vocabs)

	# Model
	model = harvard_nlp_transformer.make_model_without_embeddings(generator, d_model=d_model, n_head=n_head, d_ff=d_ff, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers, dropout=dropout_prob)

	#-----
	# Embeddings
	position = harvard_nlp_transformer.PositionalEncoding(d_model, dropout=dropout_prob)
	src_emb = torch.nn.Sequential(harvard_nlp_transformer.Embeddings(d_model, max_src_vocabs), copy.deepcopy(position))
	tgt_emb = torch.nn.Sequential(harvard_nlp_transformer.Embeddings(d_model, max_tgt_vocabs), copy.deepcopy(position))

	batch_src_emb = src_emb(batch.src)  # [batch_size, seq_len, d_model]
	batch_tgt_emb = tgt_emb(batch.tgt)  # [batch_size, seq_len - 1, d_model]

	print("Embedding:", batch_src_emb.shape, batch_tgt_emb.shape)

	if False:
		def make_canonical_masks(src_shape: torch.Size, tgt_shape: torch.Size) -> typing.Tuple[torch.Tensor, torch.Tensor]:
			src_mask = torch.ones(src_shape[:2], dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, src_seq_len]
			tgt_mask = torch.ones(tgt_shape[:2], dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, tgt_seq_len]
			tgt_mask = tgt_mask & harvard_nlp_transformer.subsequent_mask(tgt_shape[1]).type_as(tgt_mask.data)  # [batch_size, tgt_seq_len, tgt_seq_len]
			return src_mask, tgt_mask

		batch_src_mask, batch_tgt_mask = make_canonical_masks(batch_src_emb.size(), batch_tgt_emb.size())

		print("Mask:", batch_src_mask.shape, batch_tgt_mask.shape)
		assert batch_src_mask.shape == batch.src_mask.shape and batch_tgt_mask.shape == batch.tgt_mask.shape
		#assert torch.all(batch_src_mask == batch.src_mask) and torch.all(batch_tgt_mask == batch.tgt_mask)

	#-----
	# When using embeddings as input
	model_outputs = model(batch_src_emb, batch_tgt_emb, batch.src_mask, batch.tgt_mask)  # [batch_size, seq_len - 1, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, seq_len - 1, d_output]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, seq_len - 1]

	print("Model output:", model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_using_external_modules_and_token_span_test():
	print("------------------------------------------------------------")

	import copy

	emb_token_span = 3

	# Data
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	if True:
		batch = harvard_nlp_transformer.BatchWithTokenSpan(emb_token_span, srcs, tgts, pad=2)
	else:
		batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

		# Construct the masks for embeddings with multiple tokens (token span) at a timestep
		batch.src_mask = batch.src_mask.repeat_interleave(emb_token_span, dim=-1)
		batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=1)
		batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=-1)

	print("Batch:", batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs, max_tgt_vocabs = torch.max(srcs).item() + 1, torch.max(tgts).item() + 1
	d_model = 512
	n_head = 8
	d_ff = 2048
	n_enc_layers = n_dec_layers = 2
	dropout_prob = 0.1

	# Embeddings
	position = harvard_nlp_transformer.PositionalEncoding(d_model, dropout=dropout_prob)
	src_emb = torch.nn.Sequential(harvard_nlp_transformer.EmbeddingsWithTokenSpan(emb_token_span, d_model, max_src_vocabs), copy.deepcopy(position))
	tgt_emb = torch.nn.Sequential(harvard_nlp_transformer.EmbeddingsWithTokenSpan(emb_token_span, d_model, max_tgt_vocabs), copy.deepcopy(position))
	# Generator
	generator = harvard_nlp_transformer.Generator(d_model, max_tgt_vocabs)

	# Model
	model = harvard_nlp_transformer.make_model_using_external_modules(src_emb, tgt_emb, generator, d_model=d_model, n_head=n_head, d_ff=d_ff, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers, dropout=dropout_prob)

	#-----
	# When using IDs as input
	model_outputs = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)  # [batch_size, (seq_len - 1) * emb_token_span, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, (seq_len - 1) * emb_token_span, d_output]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, (seq_len - 1) * emb_token_span]

	print("Model output:", model_outputs.shape, generator_outputs.shape, pred.shape)

def transformer_without_embeddings_and_using_token_span_test():
	print("------------------------------------------------------------")

	import copy

	emb_token_span = 3

	# Data
	# SOS: 0, EOS: 1, PAD: 2
	# batch_size = 3, seq_len = 10
	srcs = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 17, 18, 1], [0, 31, 32, 33, 34, 35, 36, 27, 38, 1]])  # [batch_size, seq_len]
	tgts = torch.LongTensor([[0, 11, 12, 13, 14, 15, 16, 17, 18, 1], [0, 21, 22, 23, 24, 25, 26, 27, 1, 2], [0, 31, 32, 33, 34, 35, 36, 1, 2, 2]])  # [batch_size, seq_len]
	if True:
		batch = harvard_nlp_transformer.BatchWithTokenSpan(emb_token_span, srcs, tgts, pad=2)
	else:
		batch = harvard_nlp_transformer.Batch(srcs, tgts, pad=2)

		# Construct the masks for embeddings with multiple tokens (token span) at a timestep
		batch.src_mask = batch.src_mask.repeat_interleave(emb_token_span, dim=-1)
		batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=1)
		batch.tgt_mask = batch.tgt_mask.repeat_interleave(emb_token_span, dim=-1)

	print("Batch:", batch.src.shape, batch.tgt.shape, batch.tgt_y.shape, batch.src_mask.shape, batch.tgt_mask.shape)

	#-----
	max_src_vocabs, max_tgt_vocabs = torch.max(srcs).item() + 1, torch.max(tgts).item() + 1
	d_model = 512
	n_head = 8
	d_ff = 2048
	n_enc_layers = n_dec_layers = 2
	dropout_prob = 0.1

	# Generator
	generator = harvard_nlp_transformer.Generator(d_model, max_tgt_vocabs)

	# Model
	model = harvard_nlp_transformer.make_model_without_embeddings(generator, d_model=d_model, n_head=n_head, d_ff=d_ff, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers, dropout=dropout_prob)

	#-----
	# Embeddings
	position = harvard_nlp_transformer.PositionalEncoding(d_model, dropout=dropout_prob)
	src_emb = torch.nn.Sequential(harvard_nlp_transformer.EmbeddingsWithTokenSpan(emb_token_span, d_model, max_src_vocabs), copy.deepcopy(position))
	tgt_emb = torch.nn.Sequential(harvard_nlp_transformer.EmbeddingsWithTokenSpan(emb_token_span, d_model, max_tgt_vocabs), copy.deepcopy(position))

	batch_src_emb = src_emb(batch.src)  # [batch_size, seq_len * emb_token_span, d_model]
	batch_tgt_emb = tgt_emb(batch.tgt)  # [batch_size, (seq_len - 1) * emb_token_span, d_model]

	print("Embedding:", batch_src_emb.shape, batch_tgt_emb.shape)

	if False:
		def make_canonical_masks(src_shape: torch.Size, tgt_shape: torch.Size, src_token_span: int, tgt_token_span: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
			assert src_shape[1] % src_token_span == 0 and tgt_shape[1] % tgt_token_span == 0
			src_seq_len, tgt_seq_len = src_shape[1] // src_token_span, tgt_shape[1] // tgt_token_span
			src_mask = torch.ones((src_shape[0], src_seq_len), dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, src_seq_len]
			tgt_mask = torch.ones((tgt_shape[0], tgt_seq_len), dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, tgt_seq_len]
			tgt_mask = tgt_mask & harvard_nlp_transformer.subsequent_mask(tgt_seq_len).type_as(tgt_mask.data)  # [batch_size, tgt_seq_len, tgt_seq_len]
			# Construct the masks for embeddings with multiple tokens (token span) at a timestep
			src_mask = src_mask.repeat_interleave(src_token_span, dim=-1)  # [batch_size, 1, src_seq_len * src_token_span]
			tgt_mask = tgt_mask.repeat_interleave(tgt_token_span, dim=1).repeat_interleave(tgt_token_span, dim=-1)  # [batch_size, tgt_seq_len * tgt_token_span, tgt_seq_len * tgt_token_span]
			return src_mask, tgt_mask
			"""
			assert tgt_shape[1] % tgt_token_span == 0
			tgt_seq_len = tgt_shape[1] // tgt_token_span
			src_mask = torch.ones(src_shape[:2], dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, src_seq_len * src_token_span]
			tgt_mask = torch.ones((tgt_shape[0], tgt_seq_len), dtype=torch.bool).unsqueeze(-2)  # [batch_size, 1, tgt_seq_len]
			tgt_mask = tgt_mask & harvard_nlp_transformer.subsequent_mask(tgt_seq_len).type_as(tgt_mask.data)  # [batch_size, tgt_seq_len, tgt_seq_len]
			# Construct the masks for embeddings with multiple tokens (token span) at a timestep
			tgt_mask = tgt_mask.repeat_interleave(tgt_token_span, dim=1).repeat_interleave(tgt_token_span, dim=-1)  # [batch_size, tgt_seq_len * tgt_token_span, tgt_seq_len * tgt_token_span]
			return src_mask, tgt_mask
			"""

		batch_src_mask, batch_tgt_mask = make_canonical_masks(batch_src_emb.size(), batch_tgt_emb.size(), emb_token_span, emb_token_span)

		print("Mask:", batch_src_mask.shape, batch_tgt_mask.shape)
		assert batch_src_mask.shape == batch.src_mask.shape and batch_tgt_mask.shape == batch.tgt_mask.shape
		#assert torch.all(batch_src_mask == batch.src_mask) and torch.all(batch_tgt_mask == batch.tgt_mask)

	#-----
	# When using embeddings as input
	model_outputs = model(batch_src_emb, batch_tgt_emb, batch.src_mask, batch.tgt_mask)  # [batch_size, (seq_len - 1) * emb_token_span, d_model]
	generator_outputs = model.generator(model_outputs)  # [batch_size, (seq_len - 1) * emb_token_span, d_output]
	pred = generator_outputs.argmax(dim=-1)  # [batch_size, (seq_len - 1) * emb_token_span]

	print("Model output:", model_outputs.shape, generator_outputs.shape, pred.shape)

# REF [site] >> https://nlp.seas.harvard.edu/annotated-transformer/
def train_test():
	print("------------------------------------------------------------")

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
		model = harvard_nlp_transformer.make_model(V, V, n_enc_layers=2, n_dec_layers=2)

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

# REF [site] >> https://nlp.seas.harvard.edu/annotated-transformer/
def inference_test():
	test_model = harvard_nlp_transformer.make_model(11, 11, 512)
	test_model.eval()
	src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
	src_mask = torch.ones(1, 1, 10)

	memory = test_model.encode(src, src_mask)
	ys = torch.zeros(1, 1).type_as(src)

	for i in range(9):
		out = test_model.decode(
			memory, src_mask, ys, harvard_nlp_transformer.subsequent_mask(ys.size(1)).type_as(src.data)
		)
		prob = test_model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.data[0]
		ys = torch.cat(
			[ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
		)

	print("Example Untrained Model Prediction:", ys)

def main():
	# Harvard NLP transformer:
	#	https://nlp.seas.harvard.edu/annotated-transformer/
	#	https://github.com/harvardnlp/annotated-transformer
	#	https://github.com/fengxinjie/Transformer-OCR

	batch_test()

	#-----
	transformer_test()  # Uses internal embeddings in a transformer

	# When using external modules(embeddings & generators)
	transformer_using_external_modules_test()
	transformer_without_embeddings_test()  # When not performing embeddings in a transformer

	# When using embeddings with multiple tokens (token span) at a timestep like Decision Transformer
	transformer_using_external_modules_and_token_span_test()
	transformer_without_embeddings_and_using_token_span_test()  # When not performing embeddings in a transformer

	#-----
	train_test()
	inference_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
