#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import typing, collections, math, functools, copy, time
import torch, torchvision, torchtext

# REF [site] >> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()

		self.dropout = torch.nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embedding dim]
			output: [sequence length, batch size, embedding dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0)]
		return self.dropout(x)

class StandardTransformerModel(torch.nn.Module):
	def __init__(self, num_tokens: int, d_model: int, num_heads: int, dim_ff: int, num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.5):
		super().__init__()

		self.model_type = "Transformer"
		self.d_model = d_model
		self.sqrt_d_model = math.sqrt(self.d_model)
		self.is_individual_modules_used = True

		self.src_emb = torch.nn.Embedding(num_tokens, d_model)
		self.tgt_emb = torch.nn.Embedding(num_tokens, d_model)
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		if self.is_individual_modules_used:
			# NOTE [info] >> Call self.transformer_encoder & self.transformer_decoder directly to use as a transformer model.
			encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
			self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=None, enable_nested_tensor=True)
			decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
			self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=None)
		else:
			self.transformer = torch.nn.Transformer(d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
		self.generator = torch.nn.Linear(d_model, num_tokens)

		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		self.src_emb.weight.data.uniform_(-initrange, initrange)
		self.tgt_emb.weight.data.uniform_(-initrange, initrange)
		self.generator.bias.data.zero_()
		self.generator.weight.data.uniform_(-initrange, initrange)

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			src: Tensor, shape [seq_len, batch_size]
			src_mask: Tensor, shape [seq_len, seq_len]

		Returns:
			output Tensor of shape [seq_len, batch_size, num_tokens]
		"""

		src = self.src_emb(src) * self.sqrt_d_model
		src = self.pos_encoder(src)
		tgt = self.tgt_emb(tgt) * self.sqrt_d_model
		tgt = self.pos_encoder(tgt)
		if self.is_individual_modules_used:
			# REF [function] >> torch.nn.Transformer.forward().
			#memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=None, is_causal=None)
			memory = self.transformer_encoder(src, src_mask)
			#output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False)
			output = self.transformer_decoder(tgt, memory, tgt_mask)
		else:
			output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
		output = self.generator(output)
		#output = torch.nn.functional.sigmoid(output)
		#output = torch.nn.functional.logsigmoid(output)
		return output

# REF [site] >> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(torch.nn.Module):
	def __init__(self, num_tokens: int, d_model: int, num_heads: int, dim_ff: int, num_layers: int, dropout: float = 0.5):
		super().__init__()

		self.model_type = "Transformer"
		self.d_model = d_model
		self.sqrt_d_model = math.sqrt(self.d_model)

		self.emb = torch.nn.Embedding(num_tokens, d_model)
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
		self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None, enable_nested_tensor=True)
		self.decoder = torch.nn.Linear(d_model, num_tokens)

		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		self.emb.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			src: Tensor, shape [seq_len, batch_size]
			src_mask: Tensor, shape [seq_len, seq_len]

		Returns:
			output Tensor of shape [seq_len, batch_size, num_tokens]
		"""
		src = self.emb(src) * self.sqrt_d_model
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, src_mask)
		output = self.decoder(output)
		#output = torch.nn.functional.sigmoid(output)
		#output = torch.nn.functional.logsigmoid(output)
		return output

# REF [class] >> TransformerDecoderLayer class in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py.
class DecoderOnlyTransformerLayer(torch.nn.Module):
	__constants__ = ["batch_first", "norm_first"]

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				activation: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
				layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
				device=None, dtype=None) -> None:
		factory_kwargs = {"device": device, "dtype": dtype}
		super(DecoderOnlyTransformerLayer, self).__init__()
		self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
		#self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
		# Implementation of Feedforward model.
		self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = torch.nn.Dropout(dropout)
		self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		#self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm3 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = torch.nn.Dropout(dropout)
		#self.dropout2 = torch.nn.Dropout(dropout)
		self.dropout3 = torch.nn.Dropout(dropout)

		# Legacy string support for activation function.
		if isinstance(activation, str):
			self.activation = self._get_activation_fn(activation)
		else:
			self.activation = activation

	def __setstate__(self, state):
		if "activation" not in state:
			state["activation"] = torch.nn.functional.relu
		super(DecoderOnlyTransformerLayer, self).__setstate__(state)

	#def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: typing.Optional[torch.Tensor] = None, memory_mask: typing.Optional[torch.Tensor] = None,
	#			tgt_key_padding_mask: typing.Optional[torch.Tensor] = None, memory_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
	def forward(self, tgt: torch.Tensor, tgt_mask: typing.Optional[torch.Tensor] = None,
				tgt_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		# See Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

		x = tgt
		if self.norm_first:
			x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
			#x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
			x = x + self._ff_block(self.norm3(x))
		else:
			x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
			#x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
			x = self.norm3(x + self._ff_block(x))

		return x

	# Self-attention block.
	def _sa_block(self, x: torch.Tensor,
				attn_mask: typing.Optional[torch.Tensor], key_padding_mask: typing.Optional[torch.Tensor]) -> torch.Tensor:
		x = self.self_attn(x, x, x,
						attn_mask=attn_mask,
						key_padding_mask=key_padding_mask,
						need_weights=False)[0]
		return self.dropout1(x)

	# Multihead attention block.
	"""
	def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
				attn_mask: typing.Optional[torch.Tensor], key_padding_mask: typing.Optional[torch.Tensor]) -> torch.Tensor:
		x = self.multihead_attn(x, mem, mem,
								attn_mask=attn_mask,
								key_padding_mask=key_padding_mask,
								need_weights=False)[0]
		return self.dropout2(x)
	"""

	# Feed-forward block.
	def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout3(x)

	@staticmethod
	def _get_activation_fn(activation: str) -> typing.Callable[[torch.Tensor], torch.Tensor]:
		if activation == "relu":
			return torch.nn.functional.relu
		elif activation == "gelu":
			return torch.nn.functional.gelu

		raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# REF [class] >> TransformerDecoder class in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py.
class DecoderOnlyTransformerDecoder(torch.nn.Module):
	__constants__ = ["norm"]

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(DecoderOnlyTransformerDecoder, self).__init__()
		self.layers = self._get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	#def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: typing.Optional[torch.Tensor] = None,
	#			memory_mask: typing.Optional[torch.Tensor] = None, tgt_key_padding_mask: typing.Optional[torch.Tensor] = None,
	#			memory_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
	def forward(self, tgt: torch.Tensor, tgt_mask: typing.Optional[torch.Tensor] = None,
				tgt_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		output = tgt

		for mod in self.layers:
			"""
			output = mod(output, memory, tgt_mask=tgt_mask,
						memory_mask=memory_mask,
						tgt_key_padding_mask=tgt_key_padding_mask,
						memory_key_padding_mask=memory_key_padding_mask)
			"""
			output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

		if self.norm is not None:
			output = self.norm(output)

		return output

	@staticmethod
	def _get_clones(module, N):
		return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# REF [paper] >> "Improving Language Understanding by Generative Pre-Training", 2018 (GPT).
class DecoderOnlyTransformerModel(torch.nn.Module):
	def __init__(self, num_tokens: int, d_model: int, num_heads: int, dim_ff: int, num_layers: int, dropout: float = 0.5):
		super().__init__()

		self.model_type = "Transformer"
		self.d_model = d_model
		self.sqrt_d_model = math.sqrt(self.d_model)

		self.emb = torch.nn.Embedding(num_tokens, d_model)
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		if False:
			# NOTE [info] >> torch.nn.TransformerDecoder.forward() takes both tgt and memory as arguments. 
			#decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
			#self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=None)
			# NOTE [info] >> DecoderOnlyTransformerDecoder.forward() is modifed to take tgt only as an argument. 
			decoder_layer = DecoderOnlyTransformerLayer(d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
			self.transformer_decoder = DecoderOnlyTransformerDecoder(decoder_layer, num_layers=num_layers, norm=None)
		else:
			# NOTE [info] >> torch.nn.TransformerEncoder.forward() takes src only as an argument. 
			encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=dim_ff, dropout=dropout, activation=torch.nn.functional.relu, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
			self.transformer_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None, enable_nested_tensor=True)
		self.generator = torch.nn.Linear(d_model, num_tokens)

		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		self.emb.weight.data.uniform_(-initrange, initrange)
		self.generator.bias.data.zero_()
		self.generator.weight.data.uniform_(-initrange, initrange)

	def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
		src = self.emb(src) * self.sqrt_d_model
		src = self.pos_encoder(src)
		output = self.transformer_decoder(src, src_mask)
		output = self.generator(output)
		#output = torch.nn.functional.sigmoid(output)
		#output = torch.nn.functional.logsigmoid(output)
		return output

# REF [function] >> transformer_tutorial().
def standard_transformer_test():
	# Load and batch data.
	train_iter = torchtext.datasets.WikiText2(split="train")
	tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
	#vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
	vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>", "<sos>", "<eos>"])
	vocab.set_default_index(vocab["<unk>"])
	sos_id, eos_id = vocab(["<sos>", "<eos>"])
	num_affixes = 2  # SOS & EOS.

	def data_process(raw_text_iter: torch.utils.data.IterableDataset) -> torch.Tensor:
		"""Converts raw text into a flat Tensor."""
		data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
		return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

	# Train_iter was "consumed" by the process of building the vocab, so we have to create it again.
	train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
	train_data = data_process(train_iter)
	val_data = data_process(val_iter)
	test_data = data_process(test_iter)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
		"""Divides the data into bsz separate sequences, removing extra elements that wouldn't cleanly fit.

		Args:
			data: Tensor, shape [N]
			bsz: int, batch size

		Returns:
			Tensor of shape [N // bsz, bsz]
		"""
		seq_len = data.size(0) // bsz
		data = data[:seq_len * bsz]
		data = data.view(bsz, seq_len).t().contiguous()
		return data.to(device)  # [len(data) // batch size, batch size].

	batch_size = 20
	eval_batch_size = 10
	train_data = batchify(train_data, batch_size)  # Shape [seq_len, batch_size].
	val_data = batchify(val_data, eval_batch_size)
	test_data = batchify(test_data, eval_batch_size)

	# Functions to generate input and target sequence.
	bptt = 35
	def get_batch(source: torch.Tensor, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			source: Tensor, shape [full_seq_len, batch_size]
			i: int

		Returns:
			tuple (data, target), where data has shape [seq_len, batch_size] and target has shape [seq_len * batch_size]
		"""
		seq_len = min(bptt, len(source) - 1 - i)
		batch_size = source.shape[1]
		# Add SOS & EOS.
		def decorate(x):
			return torch.cat([torch.full((1, batch_size), fill_value=sos_id, device=device), x, torch.full((1, batch_size), fill_value=eos_id, device=device)])
		# NOTE [info] >> Not-so-good example for encoder-decoder transformer models.
		srcs = decorate(source[i:i + seq_len])
		tgts = decorate(source[i + 1:i + 1 + seq_len])  # One-step lookahead.
		return srcs, tgts  # [seq_len, batch size], [seq_len, batch size].

	# Initiate an instance.
	num_tokens = len(vocab)  # Size of vocabulary.
	dim_model = 200  # Embedding dimension.
	dim_ff = 200  # Dimension of the feedforward network model.
	num_encoder_layers = 2  # Number of transformer encoder layers.
	num_decoder_layers = 2  # Number of transformer decoder layers.
	num_heads = 2  # Number of heads in multi-head attention.
	dropout = 0.2  # Dropout probability.
	model = StandardTransformerModel(num_tokens, dim_model, num_heads, dim_ff, num_encoder_layers, num_decoder_layers, dropout).to(device)

	#-----
	# Run the model.
	criterion = torch.nn.CrossEntropyLoss()
	lr = 5.0  # Learning rate.
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	# attn_mask:
	#	2D mask: (L, S), where L is the target sequence length, S is the source sequence length.
	#	3D mask: (N * num_heads, L, S), where N is the batch size, L is the target sequence length, S is the source sequence length.
	#	attn_mask ensures that position i is allowed to attend the unmasked positions.
	#	If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged.
	#	If a BoolTensor is provided, positions with "True" are not allowed to attend while "False" values will be unchanged.
	#	If a FloatTensor is provided, it will be added to the attention weight.
	# A square attention mask is required because the self-attention layers in transformer encoders or decoders are only allowed to attend the earlier positions in the sequence.
	# REF [function] >> torch.nn.Transformer.generate_square_subsequent_mask().
	def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
	#def generate_square_subsequent_mask(sz: int, device: torch.device = torch.get_default_device(), dtype: torch.dtype = torch.get_default_dtype()) -> torch.Tensor:
		"""Generates an upper-triangular matrix of -inf, with zeros on diag."""
		return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
		#return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device), diagonal=1)

	def train(model: torch.nn.Module) -> None:
		model.train()  # Turn on train mode.
		total_loss = 0.0
		log_interval = 200
		start_time = time.time()
		src_mask = None
		#src_mask = torch.zeros(bptt, bptt, device=device)
		#tgt_mask = generate_square_subsequent_mask(bptt).to(device)
		#tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(bptt, device=device)
		tgt_mask = generate_square_subsequent_mask(bptt + num_affixes).to(device)

		num_batches = len(train_data) // bptt
		for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
			srcs, tgts = get_batch(train_data, i)
			seq_len = srcs.size(0)
			#if src_mask is not None and seq_len != bptt:  # Only on last batch.
			if src_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
				src_mask = src_mask[:seq_len, :seq_len]
			#if tgt_mask is not None and seq_len != bptt:  # Only on last batch.
			if tgt_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
				tgt_mask = tgt_mask[:seq_len, :seq_len]
			output = model(srcs, tgts[:-1], src_mask, tgt_mask[:-1,:-1])
			loss = criterion(output.view(-1, num_tokens), tgts[1:].reshape(-1))

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			total_loss += loss.item()
			if batch_idx % log_interval == 0 and batch_idx > 0:
				lr = scheduler.get_last_lr()[0]
				ms_per_batch = (time.time() - start_time) * 1000 / log_interval
				cur_loss = total_loss / log_interval
				ppl = math.exp(cur_loss)
				print(f"| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | "
					f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
					f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
				total_loss = 0
				start_time = time.time()

	def evaluate(model: torch.nn.Module, eval_data: torch.Tensor) -> float:
		model.eval()  # Turn on evaluation mode.
		total_loss = 0.0
		src_mask = None
		#src_mask = torch.zeros(bptt, bptt, device=device)
		#tgt_mask = generate_square_subsequent_mask(bptt).to(device)
		#tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(bptt, device=device)
		tgt_mask = generate_square_subsequent_mask(bptt + num_affixes).to(device)
		with torch.no_grad():
			for i in range(0, eval_data.size(0) - 1, bptt):
				srcs, tgts = get_batch(eval_data, i)
				seq_len = srcs.size(0)
				#if src_mask is not None and seq_len != bptt:  # Only on last batch.
				if src_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
					src_mask = src_mask[:seq_len, :seq_len]
				#if tgt_mask is not None and seq_len != bptt:  # Only on last batch.
				if tgt_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
					tgt_mask = tgt_mask[:seq_len, :seq_len]
				output = model(srcs, tgts[:-1], src_mask, tgt_mask[:-1,:-1])
				total_loss += seq_len * criterion(output.view(-1, num_tokens), tgts[1:].reshape(-1)).item()
		return total_loss / (len(eval_data) - 1)

	# Loop over epochs.
	# Save the model if the validation loss is the best we've seen so far. Adjust the learning rate after each epoch.
	best_val_loss = float("inf")
	epochs = 3
	best_model = None

	for epoch in range(1, epochs + 1):
		epoch_start_time = time.time()
		train(model)
		val_loss = evaluate(model, val_data)
		val_ppl = math.exp(val_loss)
		elapsed = time.time() - epoch_start_time
		print("-" * 89)
		print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
		print("-" * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = copy.deepcopy(model)

		scheduler.step()

	#-----
	# Evaluate the best model on the test dataset.
	test_loss = evaluate(best_model, test_data)
	test_ppl = math.exp(test_loss)
	print("=" * 89)
	print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
	print("=" * 89)

	#-----
	# Inference.

	if True:
		# Infer in the autoregressive way.

		assert model.is_individual_modules_used

		model.eval()  # Turn on evaluation mode.
		src_mask = None
		#src_mask = torch.zeros(bptt, bptt).to(device)
		sources, targets, predictions = [], [], []
		with torch.no_grad():
			for batch_idx, i in enumerate(range(0, test_data.size(0) - 1, bptt)):
				srcs, tgts = get_batch(test_data, i)
				seq_len, batch_size = srcs.shape[:2]
				#if src_mask is not None and seq_len != bptt:  # Only on last batch.
				if src_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
					src_mask = src_mask[:seq_len, :seq_len]

				x = model.src_emb(srcs) * model.sqrt_d_model
				x = model.pos_encoder(x)
				memory = model.transformer_encoder(src=x, mask=src_mask)  # [seq_len, batch_size, dim_model].

				# Initialize the input of the decoder with SOS.
				outputs = torch.full((1, batch_size), fill_value=sos_id, dtype=torch.long, device=device)
				for t in range(1, seq_len):
					x = model.tgt_emb(outputs) * model.sqrt_d_model
					x = model.pos_encoder(x)
					tgt_mask = generate_square_subsequent_mask(t).to(device)
					#tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=device)
					x = model.transformer_decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
					x = model.generator(x[-1:])  # The last item in time steps.

					outputs = torch.cat([outputs, x.argmax(dim=-1)], dim=0)  # [current step + 1, batch_size].

				sources.append(srcs.cpu())
				targets.append(tgts.cpu())
				predictions.append(outputs.cpu())
				if batch_idx >= 2:
					break
			sources = torch.hstack(sources)  # [seq_len, batch_size].
			targets = torch.hstack(targets)  # [seq_len, batch_size].
			predictions = torch.hstack(predictions)  # [seq_len, batch_size].
			print("The first result of inference:")
			print(vocab.lookup_tokens(sources[:,0].tolist()), vocab.lookup_tokens(targets[:,0].tolist()), vocab.lookup_tokens(predictions[:,0].tolist()))
	else:
		model.eval()  # Turn on evaluation mode.
		src_mask = None
		#src_mask = torch.zeros(bptt, bptt).to(device)
		#tgt_mask = generate_square_subsequent_mask(bptt).to(device)
		tgt_mask = generate_square_subsequent_mask(bptt + num_affixes).to(device)
		#tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(bptt + num_affixes, device=device)
		sources, targets, predictions = [], [], []
		with torch.no_grad():
			for batch_idx, i in enumerate(range(0, test_data.size(0) - 1, bptt)):
				srcs, tgts = get_batch(test_data, i)
				seq_len = srcs.size(0)
				#if src_mask is not None and seq_len != bptt:  # Only on last batch.
				if src_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
					src_mask = src_mask[:seq_len, :seq_len]
				#if tgt_mask is not None and seq_len != bptt:  # Only on last batch.
				if tgt_mask is not None and seq_len != bptt + num_affixes:  # Only on last batch.
					tgt_mask = tgt_mask[:seq_len, :seq_len]

				outputs = model(src=srcs, tgt=tgts[:-1], src_mask=src_mask, tgt_mask=tgt_mask[:-1,:-1])

				sources.append(srcs.cpu())
				targets.append(tgts.cpu())
				predictions.append(outputs.argmax(dim=-1).cpu())
				if batch_idx >= 2:
					break
			sources = torch.hstack(sources)  # [seq_len, batch_size].
			targets = torch.hstack(targets)  # [seq_len, batch_size].
			predictions = torch.hstack(predictions)  # [seq_len - 1, batch_size].
			print("The first result of inference:")
			print(vocab.lookup_tokens(sources[:,0].tolist()), vocab.lookup_tokens(targets[:,0].tolist()), vocab.lookup_tokens(predictions[:,0].tolist()))

# REF [site] >> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def transformer_tutorial():
	# Load and batch data.
	train_iter = torchtext.datasets.WikiText2(split="train")
	tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
	vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
	vocab.set_default_index(vocab["<unk>"])

	def data_process(raw_text_iter: torch.utils.data.IterableDataset) -> torch.Tensor:
		"""Converts raw text into a flat Tensor."""
		data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
		return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

	# Train_iter was "consumed" by the process of building the vocab, so we have to create it again.
	train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
	train_data = data_process(train_iter)
	val_data = data_process(val_iter)
	test_data = data_process(test_iter)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
		"""Divides the data into bsz separate sequences, removing extra elements that wouldn't cleanly fit.

		Args:
			data: Tensor, shape [N]
			bsz: int, batch size

		Returns:
			Tensor of shape [N // bsz, bsz]
		"""
		seq_len = data.size(0) // bsz
		data = data[:seq_len * bsz]
		data = data.view(bsz, seq_len).t().contiguous()
		return data.to(device)  # [len(data) // batch size, batch size].

	batch_size = 20
	eval_batch_size = 10
	train_data = batchify(train_data, batch_size)  # Shape [seq_len, batch_size].
	val_data = batchify(val_data, eval_batch_size)
	test_data = batchify(test_data, eval_batch_size)

	# Functions to generate input and target sequence.
	bptt = 35
	def get_batch(source: torch.Tensor, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			source: Tensor, shape [full_seq_len, batch_size]
			i: int

		Returns:
			tuple (data, target), where data has shape [seq_len, batch_size] and target has shape [seq_len * batch_size]
		"""
		seq_len = min(bptt, len(source) - 1 - i)
		data = source[i:i + seq_len]
		target = source[i + 1:i + 1 + seq_len]  # One-step lookahead.
		return data, target  # [seq_len, batch size], [seq_len, batch size].

	# Initiate an instance.
	num_tokens = len(vocab)  # Size of vocabulary.
	dim_model = 200  # Embedding dimension.
	dim_ff = 200  # Dimension of the feedforward network model.
	num_layers = 2  # Number of transformer encoder layers.
	num_heads = 2  # Number of heads in multi-head attention.
	dropout = 0.2  # Dropout probability.
	model = TransformerModel(num_tokens, dim_model, num_heads, dim_ff, num_layers, dropout).to(device)

	#-----
	# Run the model.
	criterion = torch.nn.CrossEntropyLoss()
	lr = 5.0  # Learning rate.
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	# attn_mask:
	#	2D mask: (L, S), where L is the target sequence length, S is the source sequence length.
	#	3D mask: (N * num_heads, L, S), where N is the batch size, L is the target sequence length, S is the source sequence length.
	#	attn_mask ensures that position i is allowed to attend the unmasked positions.
	#	If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged.
	#	If a BoolTensor is provided, positions with "True" are not allowed to attend while "False" values will be unchanged.
	#	If a FloatTensor is provided, it will be added to the attention weight.
	# A square attention mask is required because the self-attention layers in transformer encoders are only allowed to attend the earlier positions in the sequence.
	# REF [function] >> torch.nn.Transformer.generate_square_subsequent_mask().
	def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
		"""Generates an upper-triangular matrix of -inf, with zeros on diag."""
		return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

	def train(model: torch.nn.Module, is_decoder_only: bool = True) -> None:
		model.train()  # Turn on train mode.
		total_loss = 0.
		log_interval = 200
		start_time = time.time()
		if is_decoder_only:
			# Decoder-only transformer model (transformer model architecture in a decoder-only setup).
			src_mask = generate_square_subsequent_mask(bptt).to(device)
		else:
			# Encoder-only transformer model (transformer model architecture in an encoder-only setup).
			src_mask = None
			#src_mask = torch.zeros(bptt, bptt).to(device)

		num_batches = len(train_data) // bptt
		for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
			data, targets = get_batch(train_data, i)
			seq_len = data.size(0)
			if src_mask is not None and seq_len != bptt:  # Only on last batch.
				src_mask = src_mask[:seq_len, :seq_len]
			output = model(data, src_mask)
			loss = criterion(output.view(-1, num_tokens), targets.reshape(-1))

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			total_loss += loss.item()
			if batch_idx % log_interval == 0 and batch_idx > 0:
				lr = scheduler.get_last_lr()[0]
				ms_per_batch = (time.time() - start_time) * 1000 / log_interval
				cur_loss = total_loss / log_interval
				ppl = math.exp(cur_loss)
				print(f"| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | "
					f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
					f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
				total_loss = 0
				start_time = time.time()

	def evaluate(model: torch.nn.Module, eval_data: torch.Tensor, is_decoder_only: bool = True) -> float:
		model.eval()  # Turn on evaluation mode.
		total_loss = 0.0
		if is_decoder_only:
			# Decoder-only transformer model (transformer model architecture in a decoder-only setup).
			src_mask = generate_square_subsequent_mask(bptt).to(device)
		else:
			# Encoder-only transformer model (transformer model architecture in an encoder-only setup).
			src_mask = None
			#src_mask = torch.zeros(bptt, bptt).to(device)
		with torch.no_grad():
			for i in range(0, eval_data.size(0) - 1, bptt):
				data, targets = get_batch(eval_data, i)
				seq_len = data.size(0)
				if src_mask is not None and seq_len != bptt:  # Only on last batch.
					src_mask = src_mask[:seq_len, :seq_len]
				output = model(data, src_mask)
				total_loss += seq_len * criterion(output.view(-1, num_tokens), targets.reshape(-1)).item()
		return total_loss / (len(eval_data) - 1)

	# Loop over epochs.
	# Save the model if the validation loss is the best we've seen so far. Adjust the learning rate after each epoch.
	best_val_loss = float("inf")
	epochs = 3
	best_model = None
	is_decoder_only = True

	for epoch in range(1, epochs + 1):
		epoch_start_time = time.time()
		train(model, is_decoder_only)
		val_loss = evaluate(model, val_data, is_decoder_only)
		val_ppl = math.exp(val_loss)
		elapsed = time.time() - epoch_start_time
		print("-" * 89)
		print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
		print("-" * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = copy.deepcopy(model)

		scheduler.step()

	#-----
	# Evaluate the best model on the test dataset.
	test_loss = evaluate(best_model, test_data)
	test_ppl = math.exp(test_loss)
	print("=" * 89)
	print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
	print("=" * 89)

# REF [function] >> transformer_tutorial().
def decoder_based_transformer_test():
	# Load and batch data.
	train_iter = torchtext.datasets.WikiText2(split="train")
	tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
	vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
	vocab.set_default_index(vocab["<unk>"])

	def data_process(raw_text_iter: torch.utils.data.IterableDataset) -> torch.Tensor:
		"""Converts raw text into a flat Tensor."""
		data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
		return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

	# Train_iter was "consumed" by the process of building the vocab, so we have to create it again.
	train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
	train_data = data_process(train_iter)
	val_data = data_process(val_iter)
	test_data = data_process(test_iter)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
		"""Divides the data into bsz separate sequences, removing extra elements that wouldn't cleanly fit.

		Args:
			data: Tensor, shape [N]
			bsz: int, batch size

		Returns:
			Tensor of shape [N // bsz, bsz]
		"""
		seq_len = data.size(0) // bsz
		data = data[:seq_len * bsz]
		data = data.view(bsz, seq_len).t().contiguous()
		return data.to(device)  # [len(data) // batch size, batch size].

	batch_size = 20
	eval_batch_size = 10
	train_data = batchify(train_data, batch_size)  # Shape [seq_len, batch_size].
	val_data = batchify(val_data, eval_batch_size)
	test_data = batchify(test_data, eval_batch_size)

	# Functions to generate input and target sequence.
	bptt = 35
	def get_batch(source: torch.Tensor, i: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			source: Tensor, shape [full_seq_len, batch_size]
			i: int

		Returns:
			tuple (data, target), where data has shape [seq_len, batch_size] and target has shape [seq_len * batch_size]
		"""
		seq_len = min(bptt, len(source) - 1 - i)
		data = source[i:i + seq_len]
		target = source[i + 1:i + 1 + seq_len]  # One-step lookahead.
		return data, target  # [seq_len, batch size], [seq_len, batch size].

	# Initiate an instance.
	num_tokens = len(vocab)  # Size of vocabulary.
	dim_model = 200  # Embedding dimension.
	dim_ff = 200  # Dimension of the feedforward network model.
	num_layers = 2  # Number of transformer decoder layers.
	num_heads = 2  # Number of heads in multi-head attention.
	dropout = 0.2  # Dropout probability.
	model = DecoderOnlyTransformerModel(num_tokens, dim_model, num_heads, dim_ff, num_layers, dropout).to(device)

	#-----
	# Run the model.
	criterion = torch.nn.CrossEntropyLoss()
	lr = 5.0  # Learning rate.
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	# attn_mask:
	#	2D mask: (L, S), where L is the target sequence length, S is the source sequence length.
	#	3D mask: (N * num_heads, L, S), where N is the batch size, L is the target sequence length, S is the source sequence length.
	#	attn_mask ensures that position i is allowed to attend the unmasked positions.
	#	If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged.
	#	If a BoolTensor is provided, positions with "True" are not allowed to attend while "False" values will be unchanged.
	#	If a FloatTensor is provided, it will be added to the attention weight.
	# A square attention mask is required because the self-attention layers in transformer decoders are only allowed to attend the earlier positions in the sequence.
	# REF [function] >> torch.nn.Transformer.generate_square_subsequent_mask().
	def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
		"""Generates an upper-triangular matrix of -inf, with zeros on diag."""
		return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

	def train(model: torch.nn.Module, is_decoder_only: bool = True) -> None:
		model.train()  # Turn on train mode.
		total_loss = 0.
		log_interval = 200
		start_time = time.time()
		if is_decoder_only:
			# Decoder-only transformer model (transformer model architecture in a decoder-only setup).
			src_mask = generate_square_subsequent_mask(bptt).to(device)
		else:
			# Encoder-only transformer model (transformer model architecture in an encoder-only setup).
			src_mask = None
			#src_mask = torch.zeros(bptt, bptt).to(device)

		num_batches = len(train_data) // bptt
		for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
			data, targets = get_batch(train_data, i)
			seq_len = data.size(0)
			if src_mask is not None and seq_len != bptt:  # Only on last batch.
				src_mask = src_mask[:seq_len, :seq_len]
			output = model(data, src_mask)
			loss = criterion(output.view(-1, num_tokens), targets.reshape(-1))

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			total_loss += loss.item()
			if batch_idx % log_interval == 0 and batch_idx > 0:
				lr = scheduler.get_last_lr()[0]
				ms_per_batch = (time.time() - start_time) * 1000 / log_interval
				cur_loss = total_loss / log_interval
				ppl = math.exp(cur_loss)
				print(f"| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | "
					f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
					f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
				total_loss = 0
				start_time = time.time()

	def evaluate(model: torch.nn.Module, eval_data: torch.Tensor, is_decoder_only: bool = True) -> float:
		model.eval()  # Turn on evaluation mode.
		total_loss = 0.0
		if is_decoder_only:
			# Decoder-only transformer model (transformer model architecture in a decoder-only setup).
			src_mask = generate_square_subsequent_mask(bptt).to(device)
		else:
			# Encoder-only transformer model (transformer model architecture in an encoder-only setup).
			src_mask = None
			#src_mask = torch.zeros(bptt, bptt).to(device)
		with torch.no_grad():
			for i in range(0, eval_data.size(0) - 1, bptt):
				data, targets = get_batch(eval_data, i)
				seq_len = data.size(0)
				if src_mask is not None and seq_len != bptt:  # Only on last batch.
					src_mask = src_mask[:seq_len, :seq_len]
				output = model(data, src_mask)
				total_loss += seq_len * criterion(output.view(-1, num_tokens), targets.reshape(-1)).item()
		return total_loss / (len(eval_data) - 1)

	# Loop over epochs.
	# Save the model if the validation loss is the best we've seen so far. Adjust the learning rate after each epoch.
	best_val_loss = float("inf")
	epochs = 3
	best_model = None
	is_decoder_only = True

	for epoch in range(1, epochs + 1):
		epoch_start_time = time.time()
		train(model, is_decoder_only)
		val_loss = evaluate(model, val_data, is_decoder_only)
		val_ppl = math.exp(val_loss)
		elapsed = time.time() - epoch_start_time
		print("-" * 89)
		print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
		print("-" * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = copy.deepcopy(model)

		scheduler.step()

	#-----
	# Evaluate the best model on the test dataset.
	test_loss = evaluate(best_model, test_data)
	test_ppl = math.exp(test_loss)
	print("=" * 89)
	print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
	print("=" * 89)

# REF [site] >> https://pytorch.org/hub/huggingface_pytorch-transformers/
def huggingface_pytorch_transformers_test():
	import transformers

	# Tokenizer.
	tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-base-uncased")  # Download vocabulary from S3 and cache.
	#tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "./test/bert_saved_model/")  # E.g. tokenizer was saved using "save_pretrained('./test/saved_model/')".

	#-----
	# Models.
	model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-uncased")  # Download model and configuration from S3 and cache.
	#model = torch.hub.load("huggingface/pytorch-transformers", "model", "./test/bert_model/")  # E.g. model was saved using "save_pretrained('./test/saved_model/')".
	model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-uncased", output_attentions=True)  # Update configuration during loading.
	assert model.config.output_attentions == True

	# Loading from a TF checkpoint file instead of a PyTorch model (slower).
	config = transformers.AutoConfig.from_json_file("./tf_model/bert_tf_model_config.json")
	model = torch.hub.load("huggingface/pytorch-transformers", "model", "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config)

	#-----
	# Models with a language modeling head.
	model = torch.hub.load("huggingface/transformers", "modelForCausalLM", "gpt2")  # Download model and configuration from huggingface.co and cache.
	model = torch.hub.load("huggingface/transformers", "modelForCausalLM", "./test/saved_model/")  # E.g. model was saved using "save_pretrained('./test/saved_model/')".
	model = torch.hub.load("huggingface/transformers", "modelForCausalLM", "gpt2", output_attentions=True)  # Update configuration during loading.
	assert model.config.output_attentions == True

	# Loading from a TF checkpoint file instead of a PyTorch model (slower).
	config = transformers.AutoConfig.from_pretrained("./tf_model/gpt_tf_model_config.json")
	model = torch.hub.load("huggingface/transformers", "modelForCausalLM", "./tf_model/gpt_tf_checkpoint.ckpt.index", from_tf=True, config=config)

	#-----
	# Models with a sequence classification head.
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForSequenceClassification", "bert-base-uncased")  # Download model and configuration from S3 and cache.
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForSequenceClassification", "./test/bert_model/")  # E.g. model was saved using "save_pretrained('./test/saved_model/')".
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForSequenceClassification", "bert-base-uncased", output_attention=True)  # Update configuration during loading.
	assert model.config.output_attention == True

	# Loading from a TF checkpoint file instead of a PyTorch model (slower).
	config = transformers.AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForSequenceClassification", "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config)

	#-----
	# Models with a question answering head.
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForQuestionAnswering", "bert-base-uncased")  # Download model and configuration from S3 and cache.
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForQuestionAnswering", "./test/bert_model/")  # E.g. model was saved using "save_pretrained('./test/saved_model/')".
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForQuestionAnswering", "bert-base-uncased", output_attention=True)  # Update configuration during loading.
	assert model.config.output_attention == True

	# Loading from a TF checkpoint file instead of a PyTorch model (slower).
	config = transformers.AutoConfig.from_json_file("./tf_model/bert_tf_model_config.json")
	model = torch.hub.load("huggingface/pytorch-transformers", "modelForQuestionAnswering", "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config)

	#-----
	# Configuaration.
	config = torch.hub.load("huggingface/pytorch-transformers", "config", "bert-base-uncased")  # Download configuration from S3 and cache.
	config = torch.hub.load("huggingface/pytorch-transformers", "config", "./test/bert_saved_model/")  # E.g. config (or model) was saved using "save_pretrained('./test/saved_model/')".
	config = torch.hub.load("huggingface/pytorch-transformers", "config", "./test/bert_saved_model/my_configuration.json")
	config = torch.hub.load("huggingface/pytorch-transformers", "config", "bert-base-uncased", output_attention=True, foo=False)
	assert config.output_attention == True
	config, unused_kwargs = torch.hub.load("huggingface/pytorch-transformers", "config", "bert-base-uncased", output_attention=True, foo=False, return_unused_kwargs=True)
	assert config.output_attention == True
	assert unused_kwargs == {"foo": False}

	# Using the configuration with a model.
	config = torch.hub.load("huggingface/pytorch-transformers", "config", "bert-base-uncased")
	config.output_attentions = True
	config.output_hidden_states = True
	model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-uncased", config=config)
	# Model will now output attentions and hidden states as well.

	#--------------------
	# Example usage.

	# Tokenize the input.
	tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-base-cased")

	text_1 = "Who was Jim Henson ?"
	text_2 = "Jim Henson was a puppeteer"

	# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end).
	indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

	#-----
	# Using BertModel to encode the input sentence in a sequence of last layer hidden-states.

	# Define sentence A and B indices associated to 1st and 2nd sentences (see paper).
	segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

	# Convert inputs to PyTorch tensors.
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = torch.tensor([indexed_tokens])

	model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-cased")

	with torch.no_grad():
		encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)

	#-----
	# Using ModelForMaskedLM to predict a masked token with BERT.

	# Mask a token that we will try to predict back with 'BertForMaskedLM'.
	masked_index = 8
	indexed_tokens[masked_index] = tokenizer.mask_token_id
	tokens_tensor = torch.tensor([indexed_tokens])

	masked_lm_model = torch.hub.load("huggingface/pytorch-transformers", "modelForMaskedLM", "bert-base-cased")

	with torch.no_grad():
		predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

	# Get the predicted token.
	predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
	assert predicted_token == "Jim"

	#-----
	# Using ModelForQuestionAnswering to do question answering with BERT.

	question_answering_model = torch.hub.load("huggingface/pytorch-transformers", "modelForQuestionAnswering", "bert-large-uncased-whole-word-masking-finetuned-squad")
	question_answering_tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-large-uncased-whole-word-masking-finetuned-squad")

	# The format is paragraph first and then question.
	text_1 = "Jim Henson was a puppeteer"
	text_2 = "Who was Jim Henson ?"
	indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
	segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = torch.tensor([indexed_tokens])

	# Predict the start and end positions logits.
	with torch.no_grad():
		out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

	# Get the highest prediction.
	answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
	assert answer == "puppeteer"

	# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training).
	start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
	multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)

	#-----
	# Using modelforsequenceclassification to do paraphrase classification with BERT.

	sequence_classification_model = torch.hub.load("huggingface/pytorch-transformers", "modelForSequenceClassification", "bert-base-cased-finetuned-mrpc")
	sequence_classification_tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-base-cased-finetuned-mrpc")

	text_1 = "Jim Henson was a puppeteer"
	text_2 = "Who was Jim Henson ?"
	indexed_tokens = sequence_classification_tokenizer.encode(text_1, text_2, add_special_tokens=True)
	segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = torch.tensor([indexed_tokens])

	# Predict the sequence classification logits.
	with torch.no_grad():
		seq_classif_logits = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors)

	predicted_labels = torch.argmax(seq_classif_logits[0]).item()
	assert predicted_labels == 0  # In MRPC dataset this means the two sentences are not paraphrasing each other.

	# Or get the sequence classification loss (set model to train mode before if used for training).
	labels = torch.tensor([1])
	seq_classif_loss = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors, labels=labels)

# REF [function] >> pre_trained_models_example() in ./pytorch_model.py
def vit_test():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# ViT-B/16 (Base):
	#	ViT_B_16_Weights.DEFAULT, ViT_B_16_Weights.IMAGENET1K_V1, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.
	# ViT-B/32 (Base):
	#	ViT_B_32_Weights.DEFAULT, ViT_B_32_Weights.IMAGENET1K_V1.
	# ViT-L/16 (Large):
	#	ViT_L_16_Weights.DEFAULT, ViT_L_16_Weights.IMAGENET1K_V1, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.
	# ViT-L/32 (Large):
	#	ViT_L_32_Weights.DEFAULT, ViT_L_32_Weights.IMAGENET1K_V1.
	# ViT-H/14 (Huge):
	#	ViT_H_14_Weights.DEFAULT, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.

	#weights = "DEFAULT"
	#weights = "IMAGENET1K_V1"
	#weights = None  # No weights.
	weights = torchvision.models.vision_transformer.ViT_B_16_Weights.DEFAULT
	#weights = torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1

	#vit = getattr(torchvision.models, "vit_b_16")(weights=weights)  # torchvision.models.vision_transformer.VisionTransformer.
	vit = torchvision.models.vit_b_16(weights=weights)  # torchvision.models.vision_transformer.VisionTransformer.
	vit.to(device)

	if False:
		# Show the model.
		print(vit)

		for name, module in vit._modules.items():
			print(f"{name}: {type(module)}.")

		assert "conv_proj" in vit._modules
		assert "encoder" in vit._modules
		assert "heads" in vit._modules

		print(f"{type(vit.conv_proj)=}.")
		print(f"{type(vit.encoder)=}.")
		print(f"{type(vit.heads)=}.")

	if True:
		preprocess = weights.transforms()
	else:
		preprocess = torchvision.transforms.Compose([
			#torchvision.transforms.ToPILImage(),
			torchvision.transforms.Resize([224, 224]),
			#torchvision.transforms.RandomCrop([224, 224]),
			#torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	imgs = torch.rand(5, 3, 512, 512)
	inputs = preprocess(imgs)
	inputs = inputs.to(device)

	vit.eval()
	with torch.no_grad():
		if True:
			outputs = vit(inputs)
		else:
			# REF [site] >> https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
			if True:
				# Reshape and permute the input tensor.
				x = vit.submodule._process_input(inputs)  # [batch size, (image_height // patch_size) * (image_width // patch_size), hidden dim].
			else:
				x = vit.conv_proj(inputs)  # [batch size, hidden dim, image_height // patch_size, image_width // patch_size].
				x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # [batch size, (image_height // patch_size) * (image_width // patch_size), hidden dim].
			# Expand the class token to the full batch.
			batch_class_token = vit.class_token.expand(x.size(0), -1, -1)
			x = torch.cat([batch_class_token, x], dim=1)  # [batch size, 1(for classification token) + (image_height // patch_size) * (image_width // patch_size), hidden dim].
			x = vit.encoder(x)  # [batch size, 1 + (image_height // patch_size) * (image_width // patch_size), hidden dim].
			# Classifier "token" as used by standard language architectures.
			x = x[:, 0]
			outputs = vit.heads(x)  # [batch size, 1000].

	print(f"{outputs.shape=}.")

# REF [site] >> https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
class MyVisionTransformer(torch.nn.Module):
	"""Vision Transformer as per https://arxiv.org/abs/2010.11929."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		num_layers: int,
		num_heads: int,
		hidden_dim: int,
		mlp_dim: int,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		num_classes: int = 1000,
		representation_size: typing.Optional[int] = None,
		norm_layer: typing.Callable[..., torch.nn.Module] = functools.partial(torch.nn.LayerNorm, eps=1e-6),
		conv_stem_configs: typing.Optional[typing.List[torchvision.models.vision_transformer.ConvStemConfig]] = None,
	):
		super().__init__()
		torchvision.utils._log_api_usage_once(self)
		torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
		self.image_size = image_size
		self.patch_size = patch_size
		self.hidden_dim = hidden_dim
		self.mlp_dim = mlp_dim
		self.attention_dropout = attention_dropout
		self.dropout = dropout
		self.num_classes = num_classes
		self.representation_size = representation_size
		self.norm_layer = norm_layer

		if conv_stem_configs is not None:
			# As per https://arxiv.org/abs/2106.14881
			seq_proj = torch.nn.Sequential()
			prev_channels = 3
			for i, conv_stem_layer_config in enumerate(conv_stem_configs):
				seq_proj.add_module(
					f"conv_bn_relu_{i}",
					torchvision.models.ops.misc.Conv2dNormActivation(
						in_channels=prev_channels,
						out_channels=conv_stem_layer_config.out_channels,
						kernel_size=conv_stem_layer_config.kernel_size,
						stride=conv_stem_layer_config.stride,
						norm_layer=conv_stem_layer_config.norm_layer,
						activation_layer=conv_stem_layer_config.activation_layer,
					),
				)
				prev_channels = conv_stem_layer_config.out_channels
			seq_proj.add_module(
				"conv_last", torch.nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
			)
			self.conv_proj: torch.nn.Module = seq_proj
		else:
			self.conv_proj = torch.nn.Conv2d(
				in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
			)

		seq_length = (image_size // patch_size) ** 2

		"""
		# Add a class token
		self.class_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
		seq_length += 1
		"""

		self.encoder = torchvision.models.vision_transformer.Encoder(
			seq_length,
			num_layers,
			num_heads,
			hidden_dim,
			mlp_dim,
			dropout,
			attention_dropout,
			norm_layer,
		)
		self.seq_length = seq_length

		"""
		heads_layers: collections.OrderedDict[str, torch.nn.Module] = collections.OrderedDict()
		if representation_size is None:
			heads_layers["head"] = torch.nn.Linear(hidden_dim, num_classes)
		else:
			heads_layers["pre_logits"] = torch.nn.Linear(hidden_dim, representation_size)
			heads_layers["act"] = torch.nn.Tanh()
			heads_layers["head"] = torch.nn.Linear(representation_size, num_classes)

		self.heads = torch.nn.Sequential(heads_layers)
		"""

		if isinstance(self.conv_proj, torch.nn.Conv2d):
			# Init the patchify stem
			fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
			torch.nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
			if self.conv_proj.bias is not None:
				torch.nn.init.zeros_(self.conv_proj.bias)
		elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, torch.nn.Conv2d):
			# Init the last 1x1 conv of the conv stem
			torch.nn.init.normal_(
				self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
			)
			if self.conv_proj.conv_last.bias is not None:
				torch.nn.init.zeros_(self.conv_proj.conv_last.bias)

		"""
		if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, torch.nn.Linear):
			fan_in = self.heads.pre_logits.in_features
			torch.nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
			torch.nn.init.zeros_(self.heads.pre_logits.bias)

		if isinstance(self.heads.head, torch.nn.Linear):
			torch.nn.init.zeros_(self.heads.head.weight)
			torch.nn.init.zeros_(self.heads.head.bias)
		"""

	def _process_input(self, x: torch.Tensor) -> torch.Tensor:
		n, c, h, w = x.shape
		p = self.patch_size
		torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
		torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
		n_h = h // p
		n_w = w // p

		# (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
		x = self.conv_proj(x)
		# (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
		x = x.reshape(n, self.hidden_dim, n_h * n_w)

		# (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
		# The self attention layer expects inputs in the format (N, S, E)
		# where S is the source sequence length, N is the batch size, E is the
		# embedding dimension
		x = x.permute(0, 2, 1)

		return x

	def forward(self, x: torch.Tensor):
		# Reshape and permute the input tensor
		x = self._process_input(x)
		"""
		n = x.shape[0]

		# Expand the class token to the full batch
		batch_class_token = self.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)
		"""

		x = self.encoder(x)

		"""
		# Classifier "token" as used by standard language architectures
		x = x[:, 0]

		x = self.heads(x)
		"""

		return x

def customized_vit_test():
	#image_size = 224
	image_size = 448
	patch_size = 16
	assert image_size % patch_size == 0
	weights = None
	progress = True

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	#----
	# No classification token & head
	if True:
		# ViT-Base
		model = MyVisionTransformer(
			image_size=image_size,
			patch_size=patch_size,
			num_layers=12,
			num_heads=12,
			hidden_dim=768,
			mlp_dim=3072,
		)
	elif False:
		# ViT-Large
		model = MyVisionTransformer(
			image_size=image_size,
			patch_size=patch_size,
			num_layers=24,
			num_heads=16,
			hidden_dim=1024,
			mlp_dim=4096,
		)
	elif False:
		# ViT-Huge
		model = MyVisionTransformer(
			image_size=image_size,
			patch_size=patch_size,
			num_layers=32,
			num_heads=16,
			hidden_dim=1280,
			mlp_dim=5120,
		)

	if weights:
		model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

	model.to(device)

	#-----
	inputs = torch.randn(7, 3, image_size, image_size).to(device)

	print("Inferring...")
	start_time = time.time()
	model.eval()
	with torch.no_grad():
		outputs = model(inputs)  # [batch size, (image_size // patch_size) * (image_size // patch_size), hidden dim]
	print(f"Inferred: {time.time() - start_time} secs.")
	print(f"{outputs.shape=}.")

# REF [site] >> https://pytorch.org/tutorials/beginner/vt_tutorial.html
def vision_transformer_tutorial():
	import requests
	import torchvision.transforms as transforms
	import timm
	from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
	from PIL import Image

	model = torch.hub.load("facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True)
	model.eval()

	transform = transforms.Compose([
		transforms.Resize(256, interpolation=3),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
	])

	img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
	img = transform(img)[None,]

	out = model(img)
	clsidx = torch.argmax(out)
	print(clsidx.item())

	#--------------------
	#  Let's see how to modify the model so it can run on iOS and Android apps.

	# Scripting DeiT.
	# To use the model on mobile, we first need to script the model.
	#	https://pytorch.org/tutorials/recipes/script_optimized.html

	model = torch.hub.load("facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True)
	model.eval()

	scripted_model = torch.jit.script(model)
	scripted_model.save("./fbdeit_scripted.pt")  # The scripted model file fbdeit_scripted.pt of size about 346MB is generated.

	# Quantizing DeiT.
	# To reduce the trained model size significantly while keeping the inference accuracy about the same, quantization can be applied to the model.
	# Thanks to the transformer model used in DeiT, we can easily apply dynamic-quantization to the model, because dynamic quantization works best for LSTM and transformer models
	#	https://pytorch.org/docs/stable/quantization.html

	# Use 'x86' for server inference (the old 'fbgemm' is still available but 'x86' is the recommended default) and ``qnnpack`` for mobile inference.
	backend = "x86"  # Replaced with ``qnnpack`` causing much worse inference speed for quantized model on this notebook.
	model.qconfig = torch.quantization.get_default_qconfig(backend)
	torch.backends.quantized.engine = backend

	quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
	scripted_quantized_model = torch.jit.script(quantized_model)
	scripted_quantized_model.save("./fbdeit_scripted_quantized.pt")  # This generates the scripted and quantized version of the model fbdeit_quantized_scripted.pt, with size about 89MB, a 74% reduction of the non-quantized model size of 346MB!

	out = scripted_quantized_model(img)
	clsidx = torch.argmax(out)
	print(clsidx.item())  # The same output 269 should be printed.

	# Optimizing DeiT.
	from torch.utils.mobile_optimizer import optimize_for_mobile

	optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
	optimized_scripted_quantized_model.save("./fbdeit_optimized_scripted_quantized.pt")

	out = optimized_scripted_quantized_model(img)
	clsidx = torch.argmax(out)
	print(clsidx.item())  # Again, the same output 269 should be printed.

	# Using Lite Interpreter.
	# Although the lite model size is comparable to the non-lite version, when running the lite version on mobile, the inference speed up is expected.

	optimized_scripted_quantized_model._save_for_lite_interpreter("./fbdeit_optimized_scripted_quantized_lite.ptl")
	ptl = torch.jit.load("./fbdeit_optimized_scripted_quantized_lite.ptl")

	#-----
	# Comparing Inference Speed.

	with torch.autograd.profiler.profile(use_cuda=False) as prof1:
		out = model(img)
	with torch.autograd.profiler.profile(use_cuda=False) as prof2:
		out = scripted_model(img)
	with torch.autograd.profiler.profile(use_cuda=False) as prof3:
		out = scripted_quantized_model(img)
	with torch.autograd.profiler.profile(use_cuda=False) as prof4:
		out = optimized_scripted_quantized_model(img)
	with torch.autograd.profiler.profile(use_cuda=False) as prof5:
		out = ptl(img)

	print("Original model: {:.2f}ms".format(prof1.self_cpu_time_total / 1000))
	print("Scripted model: {:.2f}ms".format(prof2.self_cpu_time_total / 1000))
	print("Scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total / 1000))
	print("Scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total / 1000))
	print("Lite model: {:.2f}ms".format(prof5.self_cpu_time_total / 1000))

	import pandas as pd
	import numpy as np

	df = pd.DataFrame({"Model": ["original model", "scripted model", "scripted & quantized model", "scripted & quantized & optimized model", "lite model"]})
	df = pd.concat([
		df,
		pd.DataFrame([
			["{:.2f}ms".format(prof1.self_cpu_time_total / 1000), "0%"],
			["{:.2f}ms".format(prof2.self_cpu_time_total / 1000), "{:.2f}%".format((prof1.self_cpu_time_total - prof2.self_cpu_time_total) / prof1.self_cpu_time_total* 100)],
			["{:.2f}ms".format(prof3.self_cpu_time_total / 1000), "{:.2f}%".format((prof1.self_cpu_time_total - prof3.self_cpu_time_total) / prof1.self_cpu_time_total* 100)],
			["{:.2f}ms".format(prof4.self_cpu_time_total / 1000), "{:.2f}%".format((prof1.self_cpu_time_total - prof4.self_cpu_time_total) / prof1.self_cpu_time_total* 100)],
			["{:.2f}ms".format(prof5.self_cpu_time_total / 1000), "{:.2f}%".format((prof1.self_cpu_time_total - prof5.self_cpu_time_total) / prof1.self_cpu_time_total* 100)]
		], columns=["Inference Time", "Reduction"])
	], axis=1)

	print(df)

def main():
	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/transformer_test.py

	# Transformer layers:
	#	torch.nn.Transformer
	#	torch.nn.TransformerEncoder
	#	torch.nn.TransformerDecoder
	#	torch.nn.TransformerEncoderLayer
	#	torch.nn.TransformerDecoderLayer

	# REF [site] >> https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

	#-----
	# Standard transformer model.
	#	Encoder-decoder transformer model = transformer model architecture in an encoder-decoder setup.

	standard_transformer_test()
	#transformer_tutorial()  # Transformer encoder (encoder) + linear model (decoder).

	# Encoder-only or decoder-only transformer model.
	#	Encoder-only transformer model = transformer model architecture in an encoder-only setup.
	#	Decoder-only transformer model = transformer model architecture in a decoder-only setup.

	#decoder_based_transformer_test()  # Encoder-only or decoder-only transformer model.

	#-----
	# PyTorch-Transformers.

	#huggingface_pytorch_transformers_test()

	#--------------------
	# Vision.

	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/vit_test.py

	#vit_test()  # For image classification.
	#customized_vit_test()  # No classification token & head.

	#vision_transformer_tutorial()  # Data-efficient Image Transformers (DeiT). Scripting + quantization + optimization.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
