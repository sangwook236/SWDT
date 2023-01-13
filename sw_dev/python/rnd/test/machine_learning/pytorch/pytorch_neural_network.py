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

	# Visualizing Attention.
	output_words, attentions = evaluate(encoder, decoder, dataset.tensorFromSentence(dataset.input_lang, "je suis trop froid ."), dataset.output_lang, SOS_token, EOS_token, max_length=MAX_LENGTH, device=device)
	plt.matshow(attentions.numpy())

	evaluateAndShowAttention("elle a cinq ans de moins que moi .")
	evaluateAndShowAttention("elle est trop petit .")
	evaluateAndShowAttention("je ne crains pas de mourir .")
	evaluateAndShowAttention("c est un jeune directeur plein de talent .")

def main():
	#lenet_example()
	seq2seq_translation_tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
