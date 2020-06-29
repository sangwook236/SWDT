#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://opennmt.net/
#	https://github.com/OpenNMT/OpenNMT-py

import torch
import onmt
import onmt.translate
import onmt.utils.parse

# REF [site] >> https://opennmt.net/OpenNMT-py/Library.html
def library_example():
	preprocessed_data_dir_path = './data'

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	gpu = 0 if torch.cuda.is_available() else -1

	#--------------------
	# Prepare data.

	# Load in the vocabulary for the model of interest.
	vocab_fields = torch.load(preprocessed_data_dir_path + '/demo.vocab.pt')
	train_data_file = preprocessed_data_dir_path + '/demo.train.0.pt'
	valid_data_file = preprocessed_data_dir_path+ '/demo.valid.0.pt'

	src_text_field = vocab_fields['src'].base_field
	src_vocab = src_text_field.vocab
	src_padding = src_vocab.stoi[src_text_field.pad_token]

	tgt_text_field = vocab_fields['tgt'].base_field
	tgt_vocab = tgt_text_field.vocab
	tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

	train_iter = onmt.inputters.inputter.DatasetLazyIter(
		dataset_paths=[train_data_file], fields=vocab_fields,
		batch_size=50, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
		device=device, is_train=True, repeat=True
	)
	valid_iter = onmt.inputters.inputter.DatasetLazyIter(
		dataset_paths=[valid_data_file], fields=vocab_fields,
		batch_size=10, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
		device=device, is_train=False, repeat=False
	)

	#--------------------
	# Build a model.
	emb_size = 100
	rnn_size = 500

	# Specify the core model.
	encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab), word_padding_idx=src_padding)
	encoder = onmt.encoders.RNNEncoder(
		hidden_size=rnn_size, num_layers=1, bidirectional=True,
		rnn_type='LSTM', embeddings=encoder_embeddings
	)

	decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab), word_padding_idx=tgt_padding)
	decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
		hidden_size=rnn_size, num_layers=1, bidirectional_encoder=True, 
		rnn_type='LSTM', embeddings=decoder_embeddings
	)

	model = onmt.models.model.NMTModel(encoder, decoder)
	model.to(device)

	# Specify the tgt word generator and loss computation module.
	model.generator = torch.nn.Sequential(
		torch.nn.Linear(rnn_size, len(tgt_vocab)),
		torch.nn.LogSoftmax(dim=-1)
	).to(device)

	loss = onmt.utils.loss.NMTLossCompute(
		criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction='sum'),
		generator=model.generator
	)

	#--------------------
	# Set up an optimizer.
	lr = 1
	torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=2)

	#--------------------
	# Train.

	# Keeping track of the output requires a report manager.
	report_manager = onmt.utils.ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
	trainer = onmt.Trainer(
		model=model, train_loss=loss, valid_loss=loss,
		optim=optim, report_manager=report_manager
	)
	trainer.train(
		train_iter=train_iter, train_steps=400,
		valid_iter=valid_iter, valid_steps=200
	)

	#--------------------
	# Load up the translation functions.
	src_reader = onmt.inputters.str2reader['text']
	tgt_reader = onmt.inputters.str2reader['text']
	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0., length_penalty='avg', coverage_penalty='none')
	# Decoding strategy:
	#	Greedy search, if beam_size = 1.
	#	Beam search, otherwise.
	translator = onmt.translate.Translator(
		model=model, fields=vocab_fields, 
		src_reader=src_reader, tgt_reader=tgt_reader, 
		global_scorer=scorer, gpu=gpu
	)
	builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file), fields=vocab_fields)

	for batch in valid_iter:
		trans_batch = translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=False)
		translations = builder.from_batch(trans_batch)
		for trans in translations:
			print(trans.log(0))

#--------------------------------------------------------------------

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/preprocess.py
def preprocess_test():
	# REF [site] >> https://opennmt.net/OpenNMT-py/options/preprocess.html
	if False:
		parser = onmt.utils.parse.ArgumentParser(description='preprocess.py')

		onmt.opts.config_opts(parser)
		onmt.opts.preprocess_opts(parser)

		opt = parser.parse_args()

		onmt.utils.parse.ArgumentParser.validate_preprocess_args(opt)
	else:
		import argparse

		opt = argparse.Namespace()

		opt.config = None
		opt.data_type = 'img'
		opt.dynamic_dict = False
		opt.features_vocabs_prefix = ''
		opt.filter_valid = False
		opt.image_channel_size = 1
		opt.log_file = ''
		opt.log_file_level = '0'
		opt.lower = False
		opt.max_shard_size = 0
		opt.num_threads = 1
		opt.overwrite = False
		opt.report_every = 100000
		opt.sample_rate = 16000
		opt.save_config = None
		opt.save_data = 'data/im2text/demo'
		opt.seed = 3435
		opt.shard_size = 500
		opt.share_vocab = False
		opt.shuffle = 0
		opt.src_dir = 'data/im2text/images/'
		opt.src_seq_length = 50
		opt.src_seq_length_trunc = None
		opt.src_vocab = ''
		opt.src_vocab_size = 50000
		opt.src_words_min_frequency = 0
		opt.subword_prefix = '▁'
		opt.subword_prefix_is_joiner = False
		opt.tgt_seq_length = 150
		opt.tgt_seq_length_trunc = None
		opt.tgt_vocab = ''
		opt.tgt_vocab_size = 50000
		opt.tgt_words_min_frequency = 2
		opt.train_align = [None]
		opt.train_ids = [None]
		opt.train_src = ['data/im2text/src-train.txt']
		opt.train_tgt = ['data/im2text/tgt-train.txt']
		opt.valid_align = None
		opt.valid_src = 'data/im2text/src-val.txt'
		opt.valid_tgt = 'data/im2text/tgt-val.txt'
		opt.vocab_size_multiple = 1
		opt.window = 'hamming'
		opt.window_size = 0.02
		opt.window_stride = 0.01
	print('Preprocess options:\n', opt)

	#onmt.bin.preprocess.preprocess(opt)

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/train.py
def train_test():
	# REF [site] >> https://opennmt.net/OpenNMT-py/options/train.html
	if True:
		parser = onmt.utils.parse.ArgumentParser(description='train.py')

		onmt.opts.config_opts(parser)
		onmt.opts.model_opts(parser)
		onmt.opts.train_opts(parser)

		opt = parser.parse_args()

		onmt.utils.parse.ArgumentParser.validate_train_opts(opt)
		onmt.utils.parse.ArgumentParser.update_model_opts(opt)
		onmt.utils.parse.ArgumentParser.validate_model_opts(opt)
	else:
		import argparse

		opt = argparse.Namespace()

		opt.aan_useffn = False
		opt.accum_count = [1]
		opt.accum_steps = [0]
		opt.adagrad_accumulator_init = 0
		opt.adam_beta1 = 0.9
		opt.adam_beta2 = 0.999
		opt.alignment_heads = 0
		opt.alignment_layer = -3
		opt.apex_opt_level = 'O1'
		opt.attention_dropout = [0.1]
		opt.audio_enc_pooling = '1'
		opt.average_decay = 0
		opt.average_every = 1
		opt.batch_size = 20
		opt.batch_type = 'sents'
		opt.bridge = False
		opt.brnn = None
		opt.cnn_kernel_width = 3
		opt.config = None
		opt.context_gate = None
		opt.copy_attn = False
		opt.copy_attn_force = False
		opt.copy_attn_type = 'general'
		opt.copy_loss_by_seqlength = False
		opt.coverage_attn = False
		opt.data = 'data/im2text/demo'
		opt.data_ids = [None]
		opt.data_to_noise = []
		opt.data_weights = [1]
		opt.dec_layers = 2
		opt.dec_rnn_size = 500
		opt.decay_method = 'none'
		opt.decay_steps = 10000
		opt.decoder_type = 'rnn'
		opt.dropout = [0.3]
		opt.dropout_steps = [0]
		opt.early_stopping = 0
		opt.early_stopping_criteria = None
		opt.enc_layers = 2
		opt.enc_rnn_size = 500
		opt.encoder_type = 'brnn'
		opt.epochs = 0
		opt.exp = ''
		opt.exp_host = ''
		opt.feat_merge = 'concat'
		opt.feat_vec_exponent = 0.7
		opt.feat_vec_size = -1
		opt.fix_word_vecs_dec = False
		opt.fix_word_vecs_enc = False
		opt.full_context_alignment = False
		opt.generator_function = 'softmax'
		opt.global_attention = 'general'
		opt.global_attention_function = 'softmax'
		opt.gpu_backend = 'nccl'
		opt.gpu_ranks = [0]
		opt.gpu_verbose_level = 0
		opt.gpuid = []
		opt.heads = 8
		opt.image_channel_size = 1
		opt.input_feed = 1
		opt.keep_checkpoint = -1
		opt.label_smoothing = 0.0
		opt.lambda_align = 0.0
		opt.lambda_coverage = 0.0
		opt.layers = -1
		opt.learning_rate = 0.1
		opt.learning_rate_decay = 0.5
		opt.log_file = ''
		opt.log_file_level = '0'
		opt.loss_scale = 0
		opt.master_ip = 'localhost'
		opt.master_port = 10000
		opt.max_generator_batches = 32
		opt.max_grad_norm = 20.0
		opt.max_relative_positions = 0
		opt.model_dtype = 'fp32'
		opt.model_type = 'img'
		opt.normalization = 'sents'
		opt.optim = 'sgd'
		opt.param_init = 0.1
		opt.param_init_glorot = False
		opt.pool_factor = 8192
		opt.position_encoding = False
		opt.pre_word_vecs_dec = None
		opt.pre_word_vecs_enc = None
		opt.queue_size = 40
		opt.report_every = 50
		opt.reset_optim = 'none'
		opt.reuse_copy_attn = False
		opt.rnn_size = -1
		opt.rnn_type = 'LSTM'
		opt.sample_rate = 16000
		opt.save_checkpoint_steps = 5000
		opt.save_config = None
		opt.save_model = 'demo-model'
		opt.seed = -1
		opt.self_attn_type = 'scaled-dot'
		opt.share_decoder_embeddings = False
		opt.share_embeddings = False
		opt.single_pass = False
		opt.src_noise = []
		opt.src_noise_prob = []
		opt.src_word_vec_size = 80
		opt.start_decay_steps = 50000
		opt.tensorboard = False
		opt.tensorboard_log_dir = 'runs/onmt'
		opt.tgt_word_vec_size = 80
		opt.train_from = ''  # Checkpoint.
		opt.train_steps = 100000
		opt.transformer_ff = 2048
		opt.truncated_decoder = 0
		opt.valid_batch_size = 32
		opt.valid_steps = 10000
		opt.warmup_steps = 4000
		opt.window_size = 0.02
		opt.word_vec_size = 80
		opt.world_size = 1
	print('Train options:\n', opt)

	#onmt.bin.train.train(opt)

	vocab = torch.load(opt.data + '.vocab.pt')
	fields = vocab

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/translate.py
def translate_test():
	raise NotImplementedError

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/server.py
def server_test():
	raise NotImplementedError

#--------------------------------------------------------------------

"""
NMTModel(
  (encoder): ImageEncoder(
    (layer1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (layer2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (layer3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (layer4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (layer5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (layer6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batch_norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (batch_norm2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (batch_norm3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (rnn): LSTM(512, 250, num_layers=2, dropout=0.3, bidirectional=True)
    (pos_lut): Embedding(1000, 512)
  )
  (decoder): InputFeedRNNDecoder(
    (embeddings): Embeddings(
      (make_embedding): Sequential(
        (emb_luts): Elementwise(
          (0): Embedding(1798, 80, padding_idx=1)
        )
      )
    )
    (dropout): Dropout(p=0.3, inplace=False)
    (rnn): StackedLSTM(
      (dropout): Dropout(p=0.3, inplace=False)
      (layers): ModuleList(
        (0): LSTMCell(580, 500)
        (1): LSTMCell(500, 500)
      )
    )
    (attn): GlobalAttention(
      (linear_in): Linear(in_features=500, out_features=500, bias=False)
      (linear_out): Linear(in_features=1000, out_features=500, bias=False)
    )
  )
  (generator): Sequential(
    (0): Linear(in_features=500, out_features=1798, bias=True)
    (1): Cast()
    (2): LogSoftmax()
  )
)
"""

def build_submodels(input_channel, num_classes, word_vec_size):
	bidirectional_encoder = True
	embedding_dropout = 0.3
	encoder_num_layers = 2
	encoder_rnn_size = 500
	encoder_dropout = 0.3
	decoder_rnn_type = 'LSTM'
	decoder_num_layers = 2
	decoder_hidden_size = encoder_rnn_size
	decoder_dropout = 0.3

	src_embeddings = None
	tgt_embeddings = onmt.modules.Embeddings(
		word_vec_size=word_vec_size,
		word_vocab_size=num_classes,
		word_padding_idx=1,
		position_encoding=False,
		feat_merge='concat',
		feat_vec_exponent=0.7,
		feat_vec_size=-1,
		feat_padding_idx=[],
		feat_vocab_sizes=[],
		dropout=embedding_dropout,
		sparse=False,
		fix_word_vecs=False
	)

	encoder = onmt.encoders.ImageEncoder(
		num_layers=encoder_num_layers, bidirectional=bidirectional_encoder,
		rnn_size=encoder_rnn_size, dropout=encoder_dropout, image_chanel_size=input_channel
	)
	decoder = onmt.decoders.InputFeedRNNDecoder(
		rnn_type=decoder_rnn_type, bidirectional_encoder=bidirectional_encoder,
		num_layers=decoder_num_layers, hidden_size=decoder_hidden_size,
		attn_type='general', attn_func='softmax',
		coverage_attn=False, context_gate=None,
		copy_attn=False, dropout=decoder_dropout, embeddings=tgt_embeddings,
		reuse_copy_attn=False, copy_attn_type='general'
	)
	generator = torch.nn.Sequential(
		torch.nn.Linear(in_features=decoder_hidden_size, out_features=num_classes, bias=True),
		onmt.modules.util_class.Cast(dtype=torch.float32),
		torch.nn.LogSoftmax(dim=-1)
	)
	return encoder, decoder, generator

class MyModel(torch.nn.Module):
	def __init__(self, input_channel, num_classes, word_vec_size):
		super().__init__()

		self.encoder, self.decoder, self.generator = build_submodels(input_channel, num_classes, word_vec_size)

	# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
	def forward(self, src, tgt, lengths, bptt=False, with_align=False):
		# TODO [check] >> This function is not tested.
		dec_in = tgt[:-1]  # Exclude last target from inputs.
		enc_state, memory_bank, lengths = self.encoder(src, lengths=lengths)
		if bptt is False:
			self.decoder.init_state(src, memory_bank, enc_state)
		dec_outs, attns = self.decoder(dec_in, memory_bank, memory_lengths=lengths, with_align=with_align)
		outs = self.generator(dec_outs)
		return outs, attns

# REF [site] >> https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/model_builder.py
def build_model(use_NMTModel, input_channel, num_classes, word_vec_size):
	if use_NMTModel:
		# Use onmt.models.NMTModel.
		encoder, decoder, generator = build_submodels(input_channel, num_classes, word_vec_size)
		model = onmt.models.NMTModel(encoder, decoder)
		# NOTE [info] >> The generator is not called. So It has to be called explicitly.
		#model.generator = generator
		model.add_module('generator', generator)

		return model
	else:
		return MyModel(input_channel, num_classes, word_vec_size)

def simple_example():
	use_NMTModel = False
	input_channel = 3
	num_classes = 1798
	word_vec_size = 80
	batch_size = 16
	max_time_steps = 10

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Build a model.
	model = build_model(use_NMTModel, input_channel, num_classes, word_vec_size)
	#if model: print('Model:\n{}'.format(model))

	model = model.to(device)
	if use_NMTModel: model.generator = model.generator.to(device)

	#--------------------
	inputs = torch.randn(batch_size, input_channel, 300, 300)
	outputs = torch.randint(num_classes, (batch_size, max_time_steps, 1))
	outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].
	output_lens = torch.randint(1, max_time_steps + 1, (batch_size,))

	inputs = inputs.to(device)
	outputs, output_lens = outputs.to(device), output_lens.to(device)

	with torch.no_grad():
		model_outputs, attentions = model(inputs, outputs, output_lens)  # [target length, batch size, hidden size] & [target length, batch size, source length].
		if use_NMTModel: model_outputs = model.generator(model_outputs)

	#model_outputs = model_outputs.transpose(0, 1)  # [T, B, F] -> [B, T, F].
	model_outputs = model_outputs.cpu().numpy()
	attentions = attentions['std'].cpu().numpy()

	print("Model outputs' shape =", model_outputs.shape)
	print("Attentions' shape =", attentions.shape)

def main():
	library_example()

	#--------------------
	#preprocess_test()  # Not yet implemented.
	#train_test()  # Not yet implemented.
	#translate_test()  # Not yet implemented.
	#server_test()  # Not yet implemented.

	#--------------------
	#simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
