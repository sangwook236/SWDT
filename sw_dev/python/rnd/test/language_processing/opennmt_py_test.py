#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://opennmt.net/
#	https://github.com/OpenNMT/OpenNMT-py

import torch
import onmt
import onmt.utils.parse

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/preprocess.py
def preprocess_test():
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

def main():
	#preprocess_test()  # Not yet implemented.
	train_test()  # Not yet implemented.
	#translate_test()  # Not yet implemented.
	#server_test()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
