#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random, math, re, unicodedata, time
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# 1 input image channel, 6 output channels, 3x3 square convolution kernel.
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)
		# An affine operation: y = Wx + b.
		self.fc1 = nn.Linear(16 * 6 * 6, 120)  # For 32x32 input.
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over a (2, 2) window.
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number.
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # All dimensions except the batch dimension.
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
def lenet_example():
	net = Net()
	print('net =', net)

	# You just have to define the forward function, and the backward function (where gradients are computed) is automatically defined for you using autograd.
	# You can use any of the Tensor operations in the forward function.

	# The learnable parameters of a model are returned by net.parameters()
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight.

	if False:
		#--------------------
		input = torch.randn(1, 1, 32, 32)  # For 32x32 input.
		out = net(input)
		print('out =', out)

		#--------------------
		# torch.nn only supports mini-batches.
		# The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
		# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
		# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
		
		# Zero the gradient buffers of all parameters and backprops with random gradients.
		net.zero_grad()
		out.backward(torch.randn(1, 10))

	#--------------------
	# Loss.
	
	input = torch.randn(1, 1, 32, 32)  # For 32x32 input.
	output = net(input)
	target = torch.randn(10)  # A dummy target, for example.
	target = target.view(1, -1)  # Make it the same shape as output.
	criterion = nn.MSELoss()

	loss = criterion(output, target)
	print('loss =', loss)

	print(loss.grad_fn)  # MSELoss.
	print(loss.grad_fn.next_functions[0][0])  # Linear.
	print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU.

	#--------------------
	# Back propagation.

	# To backpropagate the error all we have to do is to loss.backward().
	# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
	net.zero_grad()  # zeroes the gradient buffers of all parameters.

	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)

	loss.backward()

	print('conv1.bias.grad after backward')
	print(net.conv1.bias.grad)

	#--------------------
	# Update the weights.

	if False:
		# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD).
		#	weight = weight - learning_rate * gradient.
		learning_rate = 0.01
		for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)

	# Create your optimizer.
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	# In your training loop.
	for _ in range(5):
		optimizer.zero_grad()  # Zero the gradient buffers.
		output = net(input)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()  # Does the update.

# REF [site] >> https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
def text_sentiment_ngrams_tutorial():
	import torchtext

	# Prepare data processing pipelines.

	tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
	train_iter = torchtext.datasets.AG_NEWS(split="train")

	def yield_tokens(data_iter):
		for _, text in data_iter:
			yield tokenizer(text)

	vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
	vocab.set_default_index(vocab["<unk>"])

	print(vocab(["here", "is", "an", "example"]))

	text_pipeline = lambda x: vocab(tokenizer(x))
	label_pipeline = lambda x: int(x) - 1

	print(text_pipeline("here is the an example"))
	print(label_pipeline("10"))

	#--------------------
	# Generate data batch and iterator.

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def collate_batch(batch):
		label_list, text_list, offsets = [], [], [0]
		for (_label, _text) in batch:
			label_list.append(label_pipeline(_label))
			processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
			text_list.append(processed_text)
			offsets.append(processed_text.size(0))
		label_list = torch.tensor(label_list, dtype=torch.int64)
		offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
		text_list = torch.cat(text_list)
		return label_list.to(device), text_list.to(device), offsets.to(device)

	train_iter = torchtext.datasets.AG_NEWS(split="train")
	dataloader = torch.utils.data.DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

	#--------------------
	# Define the model.

	class TextClassificationModel(torch.nn.Module):
		def __init__(self, vocab_size, embed_dim, num_class):
			super(TextClassificationModel, self).__init__()
			self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
			self.fc = torch.nn.Linear(embed_dim, num_class)
			self.init_weights()

		def init_weights(self):
			initrange = 0.5
			self.embedding.weight.data.uniform_(-initrange, initrange)
			self.fc.weight.data.uniform_(-initrange, initrange)
			self.fc.bias.data.zero_()

		def forward(self, text, offsets):
			embedded = self.embedding(text, offsets)
			return self.fc(embedded)

	train_iter = torchtext.datasets.AG_NEWS(split="train")
	num_class = len(set([label for (label, text) in train_iter]))
	vocab_size = len(vocab)
	emsize = 64
	model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

	#--------------------
	# Define functions to train the model and evaluate results.

	def train(dataloader):
		model.train()
		total_acc, total_count = 0, 0
		log_interval = 500
		#start_time = time.time()

		for idx, (label, text, offsets) in enumerate(dataloader):
			optimizer.zero_grad()
			predicted_label = model(text, offsets)
			loss = criterion(predicted_label, label)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
			optimizer.step()
			total_acc += (predicted_label.argmax(1) == label).sum().item()
			total_count += label.size(0)
			if idx % log_interval == 0 and idx > 0:
				#elapsed = time.time() - start_time
				print("| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count))
				total_acc, total_count = 0, 0
				#start_time = time.time()

	def evaluate(dataloader):
		model.eval()
		total_acc, total_count = 0, 0

		with torch.no_grad():
			for idx, (label, text, offsets) in enumerate(dataloader):
				predicted_label = model(text, offsets)
				#loss = criterion(predicted_label, label)
				total_acc += (predicted_label.argmax(1) == label).sum().item()
				total_count += label.size(0)
		return total_acc/total_count

	#--------------------
	# Split the dataset and run the model.

	# Hyperparameters.
	EPOCHS = 10  # Epoch.
	LR = 5  # Learning rate.
	BATCH_SIZE = 64  # Batch size for training.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=LR)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
	total_accu = None
	train_iter, test_iter = torchtext.datasets.AG_NEWS()
	train_dataset = torchtext.data.functional.to_map_style_dataset(train_iter)
	test_dataset = torchtext.data.functional.to_map_style_dataset(test_iter)
	num_train = int(len(train_dataset) * 0.95)
	split_train_, split_valid_ = torch.utils.data.dataset.random_split(train_dataset, [num_train, len(train_dataset) - num_train])

	train_dataloader = torch.utils.data.DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
	valid_dataloader = torch.utils.data.DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

	for epoch in range(1, EPOCHS + 1):
		epoch_start_time = time.time()
		train(train_dataloader)
		accu_val = evaluate(valid_dataloader)
		if total_accu is not None and total_accu > accu_val:
			scheduler.step()
		else:
			total_accu = accu_val
		print("-" * 59)
		print("| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val))
		print("-" * 59)

	#--------------------
	# Evaluate the model with test dataset.

	print("Checking the results of test dataset.")
	accu_test = evaluate(test_dataloader)
	print("test accuracy {:8.3f}".format(accu_test))

	#--------------------
	# Test on a random news.

	ag_news_label = {
		1: "World",
		2: "Sports",
		3: "Business",
		4: "Sci/Tec"
	}

	def predict(text, text_pipeline):
		with torch.no_grad():
			text = torch.tensor(text_pipeline(text))
			output = model(text, torch.tensor([0]))
			return output.argmax(1).item() + 1

	ex_text_str = "MEMPHIS, Tenn. - Four days ago, Jon Rahm was \
		enduring the season's worst weather conditions on Sunday at The \
		Open on his way to a closing 75 at Royal Portrush, which \
		considering the wind and the rain was a respectable showing. \
		Thursday's first round at the WGC-FedEx St. Jude Invitational \
		was another story. With temperatures in the mid-80s and hardly any \
		wind, the Spaniard was 13 strokes better in a flawless round. \
		Thanks to his best putting performance on the PGA Tour, Rahm \
		finished with an 8-under 62 for a three-stroke lead, which \
		was even more impressive considering he'd never played the \
		front nine at TPC Southwind."

	model = model.to("cpu")

	print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])

# REF [site] >>
#	https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
#	https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015.
def torchtext_translation_tutorial():
	import io, typing
	from collections import Counter
	import torchtext

	# Download the raw data for the English and German Spacy tokenizers.
	#	python -m spacy download en
	#	python -m spacy download de

	#--------------------
	# Data processing.

	url_base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
	train_urls = ("train.de.gz", "train.en.gz")
	val_urls = ("val.de.gz", "val.en.gz")
	test_urls = ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz")

	train_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in train_urls]
	val_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in val_urls]
	test_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in test_urls]

	de_tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="de")
	en_tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en")

	def build_vocab(filepath, tokenizer):
		counter = Counter()
		with io.open(filepath, encoding="utf8") as f:
			for string_ in f:
				counter.update(tokenizer(string_))
		return torchtext.vocab.vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])

	de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
	en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

	def data_process(filepaths):
		raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
		raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
		data = []
		for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
			de_tensor_ = torch.tensor([de_vocab[token if token in de_vocab else "<unk>"] for token in de_tokenizer(raw_de)], dtype=torch.long)
			en_tensor_ = torch.tensor([en_vocab[token if token in en_vocab else "<unk>"] for token in en_tokenizer(raw_en)], dtype=torch.long)
			data.append((de_tensor_, en_tensor_))
		return data

	train_dataset = data_process(train_filepaths)
	val_dataset = data_process(val_filepaths)
	test_dataset = data_process(test_filepaths)

	print(f"#train data = {len(train_dataset)}, #validation data = {len(val_dataset)}, #test data = {len(test_dataset)}")

	PAD_IDX = de_vocab["<pad>"]
	BOS_IDX = de_vocab["<bos>"]
	EOS_IDX = de_vocab["<eos>"]
	assert PAD_IDX == en_vocab["<pad>"]
	assert BOS_IDX == en_vocab["<bos>"]
	assert EOS_IDX == en_vocab["<eos>"]

	def generate_batch(data_batch):
		de_batch, en_batch = [], []
		for (de_item, en_item) in data_batch:
			de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
			en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
		de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=PAD_IDX)
		en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_IDX)
		return de_batch, en_batch

	BATCH_SIZE = 128
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
	valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

	print(f"#train steps per epoch = {len(train_dataloader)}, #validation steps per epoch = {len(valid_dataloader)}, #test steps per epoch = {len(test_dataloader)}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	#--------------------
	# Defining our module and optimizer.

	class Encoder(torch.nn.Module):
		def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
			super().__init__()

			self.input_dim = input_dim
			self.emb_dim = emb_dim
			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim
			self.dropout = dropout

			self.embedding = torch.nn.Embedding(input_dim, emb_dim)
			self.rnn = torch.nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
			self.fc = torch.nn.Linear(enc_hid_dim * 2, dec_hid_dim)
			self.dropout = torch.nn.Dropout(dropout)

		def forward(self, src: torch.Tensor) -> typing.Tuple[torch.Tensor]:
			embedded = self.dropout(self.embedding(src))
			outputs, hidden = self.rnn(embedded)
			hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
			return outputs, hidden

	class Attention(torch.nn.Module):
		def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
			super().__init__()

			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim

			self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
			self.attn = torch.nn.Linear(self.attn_in, attn_dim)

		def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
			src_len = encoder_outputs.shape[0]
			repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
			encoder_outputs = encoder_outputs.permute(1, 0, 2)
			energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
			attention = torch.sum(energy, dim=2)
			return torch.nn.functional.softmax(attention, dim=1)

	class Decoder(torch.nn.Module):
		def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: int, attention: torch.nn.Module):
			super().__init__()

			self.emb_dim = emb_dim
			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim
			self.output_dim = output_dim
			self.dropout = dropout
			self.attention = attention

			self.embedding = torch.nn.Embedding(output_dim, emb_dim)
			self.rnn = torch.nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
			self.out = torch.nn.Linear(self.attention.attn_in + emb_dim, output_dim)
			self.dropout = torch.nn.Dropout(dropout)

		def _weighted_encoder_rep(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
			a = self.attention(decoder_hidden, encoder_outputs)
			a = a.unsqueeze(1)

			encoder_outputs = encoder_outputs.permute(1, 0, 2)
			weighted_encoder_rep = torch.bmm(a, encoder_outputs)
			weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
			return weighted_encoder_rep

		def forward(self, input: torch.Tensor, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> typing.Tuple[torch.Tensor]:
			input = input.unsqueeze(0)
			embedded = self.dropout(self.embedding(input))
			weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
			rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
			output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
			embedded = embedded.squeeze(0)
			output = output.squeeze(0)
			weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
			output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))
			return output, decoder_hidden.squeeze(0)

	class Seq2Seq(torch.nn.Module):
		def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, device: torch.device):
			super().__init__()

			self.encoder = encoder
			self.decoder = decoder
			self.device = device

		def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
			encoder_outputs, hidden = self.encoder(src)

			batch_size = src.shape[1]
			max_len = tgt.shape[0]
			tgt_vocab_size = self.decoder.output_dim
			outputs = torch.zeros(max_len, batch_size, tgt_vocab_size, device=self.device)
			output = tgt[0,:]  # First input to the decoder is the <sos> token.
			for t in range(1, max_len):
				output, hidden = self.decoder(output, hidden, encoder_outputs)
				outputs[t] = output
				teacher_force = random.random() < teacher_forcing_ratio
				top1 = output.max(1)[1]
				output = (tgt[t] if teacher_force else top1)

			return outputs

	INPUT_DIM = len(de_vocab)
	OUTPUT_DIM = len(en_vocab)
	#ENC_EMB_DIM = 256
	#DEC_EMB_DIM = 256
	#ENC_HID_DIM = 512
	#DEC_HID_DIM = 512
	#ATTN_DIM = 64
	ENC_EMB_DIM = 32
	DEC_EMB_DIM = 32
	ENC_HID_DIM = 64
	DEC_HID_DIM = 64
	ATTN_DIM = 8
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5

	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
	attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

	model = Seq2Seq(enc, dec, device).to(device)

	def init_weights(m: torch.nn.Module) -> None:
		for name, param in m.named_parameters():
			if "weight" in name:
				torch.nn.init.normal_(param.data, mean=0, std=0.01)
			else:
				torch.nn.init.constant_(param.data, 0)

	model.apply(init_weights)

	def count_parameters(model: torch.nn.Module) -> int:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"The model has {count_parameters(model):,} trainable parameters")

	#--------------------
	# Training and evaluating the model.

	optimizer = torch.optim.Adam(model.parameters())

	#PAD_IDX = en_vocab["<pad>"]
	criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

	def train(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, clip: float) -> float:
		model.train()

		epoch_loss = 0
		for _, (src, tgt) in enumerate(iterator):
			src, tgt = src.to(device), tgt.to(device)

			optimizer.zero_grad()

			output = model(src, tgt)
			output = output[1:].view(-1, output.shape[-1])
			tgt = tgt[1:].view(-1)

			loss = criterion(output, tgt)
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()

			epoch_loss += loss.item()

		return epoch_loss / len(iterator)

	def evaluate(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> float:
		model.eval()

		epoch_loss = 0
		with torch.no_grad():
			for _, (src, tgt) in enumerate(iterator):
				src, tgt = src.to(device), tgt.to(device)

				output = model(src, tgt, 0)  # Turn off teacher forcing.

				output = output[1:].view(-1, output.shape[-1])
				tgt = tgt[1:].view(-1)

				loss = criterion(output, tgt)
				epoch_loss += loss.item()

		return epoch_loss / len(iterator)

	def epoch_time(start_time: float, end_time: float) -> typing.Tuple[int, int]:
		elapsed_time = end_time - start_time
		elapsed_mins = int(elapsed_time / 60)
		elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
		return elapsed_mins, elapsed_secs

	N_EPOCHS = 10
	CLIP = 1
	#best_valid_loss = float("inf")
	for epoch in range(N_EPOCHS):
		start_time = time.time()
		train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
		valid_loss = evaluate(model, valid_dataloader, criterion)
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
		print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
		print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")

	test_loss = evaluate(model, test_dataloader, criterion)
	print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")

# REF [site] >> https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
def char_rnn_classification_tutorial():
	raise NotImplementedError

# REF [site] >> https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
def char_rnn_generation_tutorial():
	raise NotImplementedError

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS.

	def addSentence(self, sentence):
		for word in sentence.split(" "):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

class LangDataset(torch.utils.data.Dataset):
	def __init__(self, sos, eos, max_length, device="cuda"):
		super().__init__()

		self.sos, self.eos = sos, eos
		self.max_length = max_length
		self.device = device

		eng_prefixes = (
			"i am ", "i m ",
			"he is", "he s ",
			"she is", "she s ",
			"you are", "you re ",
			"we are", "we re ",
			"they are", "they re "
		)

		self.input_lang, self.output_lang, self.pairs = self.prepareData("eng", "fra", eng_prefixes, True)
		print(random.choice(self.pairs))

		self.pairs = [self.tensorsFromPair(pair) for pair in self.pairs]

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		return self.pairs[idx]

	def prepareData(self, lang1, lang2, eng_prefixes, reverse=False):
		input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
		print("Read %s sentence pairs" % len(pairs))
		pairs = self.filterPairs(pairs, eng_prefixes)
		print("Trimmed to %s sentence pairs" % len(pairs))
		print("Counting words...")
		for pair in pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, pairs

	@staticmethod
	def readLangs(lang1, lang2, reverse=False):
		print("Reading lines...")

		# Read the file and split into lines.
		lines = open("%s-%s.txt" % (lang1, lang2), encoding="utf-8").read().strip().split("\n")

		# Split every line into pairs and normalize.
		pairs = [[LangDataset.normalizeString(s) for s in l.split("\t")] for l in lines]

		# Reverse pairs, make Lang instances.
		if reverse:
			pairs = [list(reversed(p)) for p in pairs]
			input_lang = Lang(lang2)
			output_lang = Lang(lang1)
		else:
			input_lang = Lang(lang1)
			output_lang = Lang(lang2)

		return input_lang, output_lang, pairs

	# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427.
	@staticmethod
	def unicodeToAscii(s):
		return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

	# Lowercase, trim, and remove non-letter characters.
	@staticmethod
	def normalizeString(s):
		s = LangDataset.unicodeToAscii(s.lower().strip())
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
		return s

	def filterPair(self, p, eng_prefixes):
		return len(p[0].split(" ")) < self.max_length and len(p[1].split(" ")) < self.max_length and p[1].startswith(eng_prefixes)

	def filterPairs(self, pairs, eng_prefixes):
		return [pair for pair in pairs if self.filterPair(pair, eng_prefixes)]

	@staticmethod
	def indexesFromSentence(lang, sentence):
		return [lang.word2index[word] for word in sentence.split(" ")]

	def tensorFromSentence(self, lang, sentence):
		indexes = LangDataset.indexesFromSentence(lang, sentence)
		indexes.append(self.eos)
		return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

	def tensorsFromPair(self, pair):
		input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
		target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
		return (input_tensor, target_tensor)

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, device="cuda"):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.device = device

		self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
		self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, device="cuda"):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.device = device

		self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
		self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
		self.out = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttentionDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10, device="cuda"):
		super(AttentionDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length
		self.device = device

		self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
		self.attn = nn.Linear(in_features=hidden_size * 2, out_features=max_length, bias=True)
		self.attn_combine = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=True)
		self.dropout = nn.Dropout(p=dropout_p, inplace=False)
		self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
		self.out = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=self.device)

# REF [site] >> https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def seq2seq_translation_tutorial():
	def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, sos, eos, max_length=10, teacher_forcing_ratio=0.5, device="cuda"):
		encoder_hidden = encoder.initHidden()  # Set hiddens to initial values.

		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		input_length = input_tensor.size(0)
		target_length = target_tensor.size(0)

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[sos]], device=device)
		decoder_hidden = encoder_hidden

		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		loss = 0
		if use_teacher_forcing:
			# Teacher forcing: feed the target as the next input.
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
				loss += criterion(decoder_output, target_tensor[di])
				decoder_input = target_tensor[di]  # Teacher forcing.
		else:
			# Without teacher forcing: use its own predictions as the next input.
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # Detach from history as input.

				loss += criterion(decoder_output, target_tensor[di])
				if decoder_input.item() == eos:
					break

		loss.backward()

		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.item() / target_length

	def trainIters(dataset, encoder, decoder, num_epochs, sos, eos, shuffle=True, print_every=1000, plot_every=100, learning_rate=0.01, max_length=10, teacher_forcing_ratio=0.5, device="cuda"):
		def asMinutes(s):
			m = math.floor(s / 60)
			s -= m * 60
			return '%dm %ds' % (m, s)

		def timeSince(since, percent):
			now = time.time()
			s = now - since
			es = s / (percent)
			rs = es - s
			return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

		start = time.time()
		plot_losses = []
		print_loss_total = 0  # Reset every print_every.
		plot_loss_total = 0  # Reset every plot_every.

		encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
		decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
		criterion = nn.NLLLoss()

		num_examples = len(dataset)
		examples_indices = list(range(num_examples))
		if shuffle: random.shuffle(examples_indices)
		n_iters = num_epochs * num_examples
		iter = 1
		for epoch in range(num_epochs):
			for idx in examples_indices:
				input_tensor, target_tensor = dataset[idx]

				loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, sos, eos, max_length, teacher_forcing_ratio, device)
				print_loss_total += loss
				plot_loss_total += loss

				if iter % print_every == 0:
					print_loss_avg = print_loss_total / print_every
					print_loss_total = 0
					print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

				if iter % plot_every == 0:
					plot_loss_avg = plot_loss_total / plot_every
					plot_losses.append(plot_loss_avg)
					plot_loss_total = 0

				iter += 1

		def showPlot(points):
			plt.figure()
			fig, ax = plt.subplots()
			# This locator puts ticks at regular intervals.
			loc = ticker.MultipleLocator(base=0.2)
			ax.yaxis.set_major_locator(loc)
			plt.plot(points)

		showPlot(plot_losses)

	def evaluate(encoder, decoder, input_tensor, output_lang, sos, eos, max_length=10, device="cuda"):
		with torch.no_grad():
			input_length = input_tensor.size()[0]
			encoder_hidden = encoder.initHidden()  # Set hiddens to initial values.

			encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

			for ei in range(input_length):
				encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] += encoder_output[0, 0]

			decoder_input = torch.tensor([[sos]], device=device)  # SOS.
			decoder_hidden = encoder_hidden

			decoded_words = []
			decoder_attentions = torch.zeros(max_length, max_length)

			for di in range(max_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
				decoder_attentions[di] = decoder_attention.data
				topv, topi = decoder_output.data.topk(1)
				if topi.item() == eos:
					decoded_words.append('<EOS>')
					break
				else:
					decoded_words.append(output_lang.index2word[topi.item()])

				decoder_input = topi.squeeze().detach()

			return decoded_words, decoder_attentions[:di + 1]

	def evaluateRandomly(dataset, encoder, decoder, sos, eos, n=10, max_length=10, device="cuda"):
		for i in range(n):
			pair = random.choice(dataset.pairs)
			print('>', ' '.join(dataset.input_lang.index2word[idx.item()] for idx in pair[0].cpu()))
			print('=', ' '.join(dataset.output_lang.index2word[idx.item()] for idx in pair[1].cpu()))
			output_words, attentions = evaluate(encoder, decoder, pair[0], dataset.output_lang, sos, eos, max_length, device)
			output_sentence = ' '.join(output_words)
			print('<', output_sentence)
			print('')

	def showAttention(input_sentence, output_words, attentions):
		# Set up figure with colorbar.
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(attentions.numpy(), cmap='bone')
		fig.colorbar(cax)

		# Set up axes.
		ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
		ax.set_yticklabels([''] + output_words)

		# Show label at every tick.
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

		plt.show()

	def evaluateAndShowAttention(input_sentence):
		output_words, attentions = evaluate(encoder, decoder, dataset.tensorFromSentence(dataset.input_lang, input_sentence), dataset.output_lang, SOS_token, EOS_token, max_length=MAX_LENGTH, device=device)
		print('input =', input_sentence)
		print('output =', ' '.join(output_words))
		showAttention(input_sentence, output_words, attentions)

	#-----
	SOS_token = 0  # Fixed.
	EOS_token = 1  # Fixed.
	MAX_LENGTH = 10
	teacher_forcing_ratio = 0.5
	hidden_size = 256
	num_epochs = 10

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = LangDataset(SOS_token, EOS_token, max_length=MAX_LENGTH)
	print(f"len(dataset) = {len(dataset)}.")
	print(f"Input dim = {dataset.input_lang.n_words}, output dim = {dataset.output_lang.n_words}.")

	# Training and evaluating.
	encoder = EncoderRNN(dataset.input_lang.n_words, hidden_size, device=device).to(device)
	#decoder = DecoderRNN(hidden_size, dataset.output_lang.n_words, device=device).to(device)
	decoder = AttentionDecoderRNN(hidden_size, dataset.output_lang.n_words, dropout_p=0.1, device=device).to(device)

	trainIters(dataset, encoder, decoder, num_epochs, SOS_token, EOS_token, shuffle=True, print_every=5000, max_length=MAX_LENGTH, teacher_forcing_ratio=teacher_forcing_ratio, device=device)

	evaluateRandomly(dataset, encoder, decoder, SOS_token, EOS_token, max_length=MAX_LENGTH, device=device)

	# Visualizing attention.
	output_words, attentions = evaluate(encoder, decoder, dataset.tensorFromSentence(dataset.input_lang, "je suis trop froid ."), dataset.output_lang, SOS_token, EOS_token, max_length=MAX_LENGTH, device=device)
	plt.matshow(attentions.numpy())

	evaluateAndShowAttention("elle a cinq ans de moins que moi .")
	evaluateAndShowAttention("elle est trop petit .")
	evaluateAndShowAttention("je ne crains pas de mourir .")
	evaluateAndShowAttention("c est un jeune directeur plein de talent .")

def main():
	#lenet_example()

	#text_sentiment_ngrams_tutorial()
	torchtext_translation_tutorial()  # Seq2seq model.
	#char_rnn_classification_tutorial()  # Not yet implemented.
	#char_rnn_generation_tutorial()  # Not yet implemented.
	#seq2seq_translation_tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
