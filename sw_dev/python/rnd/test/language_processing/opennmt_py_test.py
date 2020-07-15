#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://opennmt.net/
#	https://github.com/OpenNMT/OpenNMT-py

import argparse, time
import torch
import torchtext
import onmt
import onmt.translate
import onmt.utils.parse

def save_model(model_filepath, model, generator):
	#torch.save(model.state_dict(), model_filepath)
	#torch.save({'state_dict': model.state_dict()}, model_filepath)
	torch.save({'model': model.state_dict(), 'generator': generator.state_dict()}, model_filepath)
	#torch.save({'model': model.state_dict(), 'generator': generator.state_dict(), 'optim': optim.state_dict()}, model_filepath)
	print('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model, generator, device='cpu'):
	"""
	loaded_data = torch.load(model_filepath, map_location=device)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	print('Loaded a model from {}.'.format(model_filepath))
	return model
	"""
	checkpoint = torch.load(model_filepath, map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['model'])
	generator.load_state_dict(checkpoint['generator'])
	#optim.load_state_dict(checkpoint['optim'])
	#opt = checkpoint['opt']
	#vocab = checkpoint['vocab']
	#epoch = checkpoint['epoch']
	print('Loaded a model from {}.'.format(model_filepath))
	return model, generator

#--------------------------------------------------------------------

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/preprocess.py
def preprocess_test():
	# REF [site] >> https://opennmt.net/OpenNMT-py/options/preprocess.html
	if False:
		parser = onmt.utils.parse.ArgumentParser(description='preprocess_test')

		onmt.opts.config_opts(parser)
		onmt.opts.preprocess_opts(parser)

		opt = parser.parse_args()
	else:
		opt = argparse.Namespace()

		opt.config = None  # Config file path (default: None).
		opt.save_config = None  # Config file save path (default: None).

		# Data.
		opt.data_type = 'img'  # Type of the source input. Options are [text|img|audio|vec]. (default: text).
		opt.train_src = ['data/im2text/src-train.txt']  # Path(s) to the training source data (default: None).
		opt.train_tgt = ['data/im2text/tgt-train.txt']  # Path(s) to the training target data (default: None).
		opt.train_align = [None]  # Path(s) to the training src-tgt alignment (default: [None]).
		opt.train_ids = [None]  # IDs to name training shards, used for corpus weighting (default: [None]).
		opt.valid_src = 'data/im2text/src-val.txt'  # Path to the validation source data (default: None).
		opt.valid_tgt = 'data/im2text/tgt-val.txt'  # Path to the validation target data (default: None).
		opt.valid_align = None  # Path(s) to the validation src-tgt alignment (default: None).
		opt.src_dir = 'data/im2text/images/'  # Source directory for image or audio files. (default: ).
		opt.save_data = 'data/im2text/demo'  # Output file for the prepared data (default: None).
		opt.max_shard_size = 0  # Deprecated use shard_size instead (default: 0).
		opt.shard_size = 500  # Divide src_corpus and tgt_corpus into smaller multiple src_copus and tgt corpus files, then build shards, each shard will have opt.shard_size samples except last shard. shard_size=0 means no segmentation shard_size>0 means segment dataset into multiple shards, each shard has shard_size samples (default: 1000000)
		opt.num_threads = 1  # Number of shards to build in parallel. (default: 1).
		opt.overwrite = False  # Overwrite existing shards if any. (default: False).

		# Vocab.
		opt.src_vocab = ''  # Path to an existing source vocabulary. Format: one word per line. (default: ).
		opt.tgt_vocab = ''  # Path to an existing target vocabulary. Format: one word per line. (default: ).
		opt.features_vocabs_prefix = ''  # Path prefix to existing features vocabularies (default: ).
		opt.src_vocab_size = 50000  # Size of the source vocabulary (default: 50000).
		opt.tgt_vocab_size = 50000  # Size of the target vocabulary (default: 50000).
		opt.vocab_size_multiple = 1  # Make the vocabulary size a multiple of this value (default: 1).
		opt.src_words_min_frequency = 0
		opt.tgt_words_min_frequency = 2
		opt.dynamic_dict = False  # Create dynamic dictionaries (default: False).
		opt.share_vocab = False  # Share source and target vocabulary (default: False).

		# Pruning.
		opt.src_seq_length = 50  # Maximum source sequence length (default: 50).
		opt.src_seq_length_trunc = None  # Truncate source sequence length. (default: None).
		opt.tgt_seq_length = 150  # Maximum target sequence length to keep. (default: 50).
		opt.tgt_seq_length_trunc = None  # Truncate target sequence length. (default: None).
		opt.lower = False  # Lowercase data (default: False).
		opt.filter_valid = False  # Filter validation data by src and/or tgt length (default: False).

		# Random.
		opt.shuffle = 0  # Shuffle data (default: 0).
		opt.seed = 3435  # Random seed (default: 3435).

		# Logging.
		opt.report_every = 100000  # Report status every this many sentences (default: 100000).
		opt.log_file = ''  # Output logs to a file under this path. (default: ).
		opt.log_file_level = '0'  # {CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET, 50, 40, 30, 20, 10, 0}.

		# Speech.
		opt.sample_rate = 16000  # Sample rate. (default: 16000).
		opt.window_size = 0.02  # Window size for spectrogram in seconds. (default: 0.02).
		opt.window_stride = 0.01  # Window stride for spectrogram in seconds. (default: 0.01).
		opt.window = 'hamming'  # Window type for spectrogram generation. (default: hamming).

		# Image.
		opt.image_channel_size = 1  # Using grayscale image can training model faster and smaller {3, 1} (default: 3).

		# Noise.
		opt.subword_prefix = '_'  # Subword prefix to build wordstart mask (default: _).
		opt.subword_prefix_is_joiner = False  # mask will need to be inverted if prefix is joiner (default: False).

	print('Preprocess options:\n{}'.format(opt))

	#------------------------------------------------------------
	#onmt.bin.preprocess.preprocess(opt)

	#------------------------------------------------------------
	# REF [function] >> preprocess() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/bin/preprocess.py.
	onmt.utils.parse.ArgumentParser.validate_preprocess_args(opt)

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/train.py
def train_test():
	# REF [site] >> https://opennmt.net/OpenNMT-py/options/train.html
	if True:
		parser = onmt.utils.parse.ArgumentParser(description='train_test')

		onmt.opts.config_opts(parser)
		onmt.opts.model_opts(parser)
		onmt.opts.train_opts(parser)

		opt = parser.parse_args()
	else:
		opt = argparse.Namespace()

		opt.config = None  # Config file path (default: None).
		opt.save_config = None  # Config file save path (default: None).

		# Model-Embeddings.
		opt.src_word_vec_size = 80  # Word embedding size for src. (default: 500).
		opt.tgt_word_vec_size = 80  # Word embedding size for tgt. (default: 500).
		opt.word_vec_size = 80  # Word embedding size for src and tgt. (default: -1).
		opt.share_decoder_embeddings = False  # Use a shared weight matrix for the input and output word embeddings in the decoder. (default: False).
		opt.share_embeddings = False  # Share the word embeddings between encoder and decoder. Need to use shared dictionary for this option. (default: False).
		opt.position_encoding = False  # Use a sin to mark relative words positions. Necessary for non-RNN style models. (default: False).

		# Model: Embedding Features.
		opt.feat_merge = 'concat'  # Merge action for incorporating features embeddings. Options [concat|sum|mlp]. (default: concat).
		opt.feat_vec_size = -1  # If specified, feature embedding sizes will be set to this. Otherwise, feat_vec_exponent will be used. (default: -1).
		opt.feat_vec_exponent = 0.7  # If -feat_merge_size is not set, feature embedding sizes will be set to N^feat_vec_exponent where N is the number of values the feature takes. (default: 0.7).

		# Model: Encoder-Decoder.
		opt.model_type = 'img'  # Type of source model to use. Allows the system to incorporate non-text inputs. Options are [text|img|audio|vec]. (default: text).
		opt.model_dtype = 'fp32'  # Data type of the model. {fp32, fp16}. (default: fp32).
		opt.encoder_type = 'brnn'  # Type of encoder layer to use. Non-RNN layers are experimental. Options are [rnn|brnn|mean|transformer|cnn]. (default: rnn).
		opt.decoder_type = 'rnn'  # Type of decoder layer to use. Non-RNN layers are experimental. Options are [rnn|transformer|cnn]. (default: rnn).
		opt.layers = -1  # Number of layers in enc/dec. (default: -1).
		opt.enc_layers = 2  # Number of layers in the encoder (default: 2).
		opt.dec_layers = 2  # Number of layers in the decoder (default: 2).
		opt.rnn_size = -1  # Size of rnn hidden states. Overwrites enc_rnn_size and dec_rnn_size (default: -1).
		opt.enc_rnn_size = 500  # Size of encoder rnn hidden states. Must be equal to dec_rnn_size except for speech-to-text. (default: 500).
		opt.dec_rnn_size = 500  # Size of decoder rnn hidden states. Must be equal to enc_rnn_size except for speech-to-text. (default: 500).
		opt.audio_enc_pooling = '1'  # The amount of pooling of audio encoder, either the same amount of pooling across all layers indicated by a single number, or different amounts of pooling per layer separated by comma. (default: 1).
		opt.cnn_kernel_width = 3  # Size of windows in the cnn, the kernel_size is (cnn_kernel_width, 1) in conv layer (default: 3).
		opt.input_feed = 1  # Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder. (default: 1).
		opt.bridge = False  # Have an additional layer between the last encoder state and the first decoder state (default: False).
		opt.rnn_type = 'LSTM'  # The gate type to use in the RNNs {LSTM, GRU, SRU} (default: LSTM).
		opt.brnn = None  # Deprecated, use 'encoder_type'. (default: None).
		opt.context_gate = None  # Type of context gate to use. Do not select for no context gate. {source, target, both} (default: None).

		# Model: Attention.
		opt.global_attention = 'general'  # The attention type to use: dotprod or general (Luong) or MLP (Bahdanau) {dot, general, mlp, none} (default: general).
		opt.global_attention_function = 'softmax'  # {softmax, sparsemax}.
		opt.self_attn_type = 'scaled-dot'  # Self attention type in Transformer decoder layer -- currently "scaled-dot" or "average" (default: scaled-dot).
		opt.max_relative_positions = 0  # Maximum distance between inputs in relative positions representations. For more detailed information, see: https://arxiv.org/pdf/1803.02155.pdf (default: 0).
		opt.heads = 8  # Number of heads for transformer self-attention (default: 8).
		opt.transformer_ff = 2048  # Size of hidden transformer feed-forward (default: 2048).
		opt.aan_useffn = False  # Turn on the FFN layer in the AAN decoder (default: False).

		# Model: Alignement.
		opt.lambda_align = 0.0  # Lambda value for alignement loss of Garg et al (2019) For more detailed information, see: https://arxiv.org/abs/1909.02074 (default: 0.0).
		opt.alignment_layer = -3  # Layer number which has to be supervised. (default: -3).
		opt.alignment_heads = 0  # N. of cross attention heads per layer to supervised with (default: 0).
		opt.full_context_alignment = False  # Whether alignment is conditioned on full target context. (default: False).

		# Generator.
		opt.copy_attn = False  # Train copy attention layer. (default: False).
		opt.copy_attn_type = 'general'  # The copy attention type to use. Leave as None to use the same as -global_attention. {dot, general, mlp, none} (default: None).
		opt.generator_function = 'softmax'  # Which function to use for generating probabilities over the target vocabulary (choices: softmax, sparsemax) (default: softmax).
		opt.copy_attn_force = False  # When available, train to copy. (default: False).
		opt.reuse_copy_attn = False  # Reuse standard attention for copy (default: False).
		opt.copy_loss_by_seqlength = False  # Divide copy loss by length of sequence (default: False).
		opt.coverage_attn = False  # Train a coverage attention layer. (default: False).
		opt.lambda_coverage = 0.0  # Lambda value for coverage loss of See et al (2017) (default: 0.0).
		opt.loss_scale = 0  # For FP16 training, the static loss scale to use. If not set, the loss scale is dynamically computed. (default: 0).
		opt.apex_opt_level = 'O1'  # For FP16 training, the opt_level to use. See https://nvidia.github.io/apex/amp.html#opt-levels. {O0, O1, O2, O3} (default: O1).

		# General.
		opt.data = 'data/im2text/demo'  # Path prefix to the ".train.pt" and ".valid.pt" file path from preprocess.py (default: None).
		opt.data_ids = [None]  # In case there are several corpora. (default: [None]).
		opt.data_weights = [1]  # Weights of different corpora, should follow the same order as in -data_ids. (default: [1]).
		opt.data_to_noise = []  # IDs of datasets on which to apply noise. (default: []).
		opt.save_model = 'demo-model'  # Model filename (the model will be saved as <save_model>_N.pt where N is the number of steps (default: model).
		opt.save_checkpoint_steps = 5000  # Save a checkpoint every X steps (default: 5000).
		opt.keep_checkpoint = -1  # Keep X checkpoints (negative: keep all) (default: -1).
		opt.gpuid = []  # Deprecated see world_size and gpu_ranks. (default: []).
		opt.gpu_ranks = [0]  # List of ranks of each process. (default: []).
		opt.world_size = 1  # Total number of distributed processes. (default: 1).
		opt.gpu_backend = 'nccl'  # Type of torch distributed backend (default: nccl).
		opt.gpu_verbose_level = 0  # Gives more info on each process per GPU. (default: 0).
		opt.master_ip = 'localhost'  # IP of master for torch.distributed training. (default: localhost).
		opt.master_port = 10000  # Port of master for torch.distributed training. (default: 10000).
		opt.queue_size = 40  # Size of queue for each process in producer/consumer (default: 40).
		opt.seed = -1  # Random seed used for the experiments reproducibility. (default: -1).

		# Initialization.
		opt.param_init = 0.1  # Parameters are initialized over uniform distribution with support (-param_init, param_init). Use 0 to not use initialization (default: 0.1).
		opt.param_init_glorot = False  # Init parameters with xavier_uniform. Required for transformer. (default: False).
		opt.train_from = ''  # If training from a checkpoint then this is the path to the pretrained model's state_dict. (default: ).
		opt.reset_optim = 'none'  # Optimization resetter when train_from. {none, all, states, keep_states} (default: none).
		opt.pre_word_vecs_enc = None  # If a valid path is specified, then this will load pretrained word embeddings on the encoder side. See README for specific formatting instructions. (default: None).
		opt.pre_word_vecs_dec = None  # If a valid path is specified, then this will load pretrained word embeddings on the decoder side. See README for specific formatting instructions. (default: None)
		opt.fix_word_vecs_enc = False  # Fix word embeddings on the encoder side. (default: False).
		opt.fix_word_vecs_dec = False  # Fix word embeddings on the decoder side. (default: False).

		# Optimization: Type.
		opt.batch_size = 20  # Maximum batch size for training (default: 64).
		opt.batch_type = 'sents'  # Batch grouping for batch_size. Standard is sents. Tokens will do dynamic batching {sents, tokens} (default: sents).
		opt.pool_factor = 8192  # Factor used in data loading and batch creations. It will load the equivalent of 'pool_factor' batches, sort them by the according 'sort_key' to produce homogeneous batches and reduce padding, and yield the produced batches in a shuffled way. Inspired by torchtext's pool mechanism. (default: 8192).
		opt.normalization = 'sents'  # Normalization method of the gradient. {sents, tokens} (default: sents).
		opt.accum_count = [1]  # Accumulate gradient this many times. Approximately equivalent to updating batch_size * accum_count batches at once. Recommended for Transformer. (default: [1]).
		opt.accum_steps = [0]  # Steps at which accum_count values change (default: [0]).
		opt.valid_steps = 10000  # Perfom validation every X steps (default: 10000).
		opt.valid_batch_size = 32  # Maximum batch size for validation (default: 32).
		opt.max_generator_batches = 32  # Maximum batches of words in a sequence to run the generator on in parallel. Higher is faster, but uses more memory. Set to 0 to disable. (default: 32).
		opt.train_steps = 100000  # Number of training steps (default: 100000).
		opt.single_pass = False  # Make a single pass over the training dataset. (default: False).
		opt.epochs = 0  # Deprecated epochs see train_steps (default: 0).
		opt.early_stopping = 0  # Number of validation steps without improving. (default: 0).
		opt.early_stopping_criteria = None  # Criteria to use for early stopping. (default: None).
		opt.optim = 'sgd'  # Optimization method. {sgd, adagrad, adadelta, adam, sparseadam, adafactor, fusedadam} (default: sgd).
		opt.adagrad_accumulator_init = 0  # Initializes the accumulator values in adagrad. Mirrors the initial_accumulator_value option in the tensorflow adagrad (use 0.1 for their default). (default: 0).
		opt.max_grad_norm = 20.0  # If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm (default: 5).
		opt.dropout = [0.3]  # Dropout probability; applied in LSTM stacks. (default: [0.3]).
		opt.attention_dropout = [0.1]  # Attention Dropout probability. (default: [0.1]).
		opt.dropout_steps = [0]  # Steps at which dropout changes. (default: [0]).
		opt.truncated_decoder = 0  # Truncated bptt. (default: 0).
		opt.adam_beta1 = 0.9  # The beta1 parameter used by Adam. Almost without exception a value of 0.9 is used in the literature, seemingly giving good results, so we would discourage changing this value from the default without due consideration. (default: 0.9).
		opt.adam_beta2 = 0.999  # The beta2 parameter used by Adam. Typically a value of 0.999 is recommended, as this is the value suggested by the original paper describing Adam, and is also the value adopted in other frameworks such as Tensorflow and Keras, i.e. see: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer or https://keras.io/optimizers/. Whereas recently the paper "Attention is All You Need" suggested a value of 0.98 for beta2, this parameter may not work well for normal models / default baselines. (default: 0.999)
		opt.label_smoothing = 0.0  # Label smoothing value epsilon. Probabilities of all non-true labels will be smoothed by epsilon / (vocab_size - 1). Set to zero to turn off label smoothing. For more detailed information, see: https://arxiv.org/abs/1512.00567 (default: 0.0).
		opt.average_decay = 0  # Moving average decay. Set to other than 0 (e.g. 1e-4) to activate. Similar to Marian NMT implementation: http://www.aclweb.org/anthology/P18-4020 For more detail on Exponential Moving Average: https://en.wikipedia.org/wiki/Moving_average (default: 0).
		opt.average_every = 1  # Step for moving average. Default is every update, if -average_decay is set. (default: 1).
		opt.src_noise = []  # {sen_shuffling, infilling, mask}.
		opt.src_noise_prob = []  # Probabilities of src_noise functions (default: []).

		# Optimization: Rate.
		opt.learning_rate = 0.1  # Starting learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001 (default: 1.0).
		opt.learning_rate_decay = 0.5  # If update_learning_rate, decay learning rate by this much if steps have gone past start_decay_steps (default: 0.5).
		opt.start_decay_steps = 50000  # Start decaying every decay_steps after start_decay_steps (default: 50000).
		opt.decay_steps = 10000  # Decay every decay_steps (default: 10000).
		opt.decay_method = 'none'  # Use a custom decay rate. (default: none).
		opt.warmup_steps = 4000  # Number of warmup steps for custom decay. (default: 4000).

		# Logging.
		opt.report_every = 50  # Print stats at this interval. (default: 50).
		opt.log_file = ''  # Output logs to a file under this path. (default: ).
		opt.log_file_level = '0'  # {CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET, 50, 40, 30, 20, 10, 0}.
		opt.exp_host = ''  # Send logs to this crayon server. (default: ).
		opt.exp = ''  # Name of the experiment for logging. (default: ).
		opt.tensorboard = False  # Use tensorboard for visualization during training. Must have the library tensorboard >= 1.14. (default: False).
		opt.tensorboard_log_dir = 'runs/onmt'  # Log directory for Tensorboard. This is also the name of the run. (default: runs/onmt).

		# Speech.
		opt.sample_rate = 16000  # Sample rate. (default: 16000).
		opt.window_size = 0.02  # Window size for spectrogram in seconds. (default: 0.02).

		# Image.
		opt.image_channel_size = 1  # Using grayscale image can training model faster and smaller {3, 1} (default: 3).

	print('Train options:\n{}'.format(opt))

	#------------------------------------------------------------
	#onmt.bin.train.train(opt)

	#------------------------------------------------------------
	# REF [function] >> train() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/bin/train.py.
	onmt.utils.parse.ArgumentParser.validate_train_opts(opt)
	#onmt.utils.parse.ArgumentParser.update_model_opts(opt)
	#onmt.utils.parse.ArgumentParser.validate_model_opts(opt)

	# REF [function] >> main() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/train_single.py.
	if opt.train_from:
		onmt.utils.logging.logger.info('Loading checkpoint from {}.'.format(opt.train_from))
		checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
		model_opt = onmt.utils.parse.ArgumentParser.ckpt_model_opts(checkpoint['opt'])
		onmt.utils.parse.ArgumentParser.update_model_opts(model_opt)
		onmt.utils.parse.ArgumentParser.validate_model_opts(model_opt)
		onmt.utils.logging.logger.info('Loading vocab from checkpoint at {}.'.format(opt.train_from))
		vocab = checkpoint['vocab']
	else:
		checkpoint = None
		model_opt = opt
		onmt.utils.parse.ArgumentParser.update_model_opts(model_opt)
		onmt.utils.parse.ArgumentParser.validate_model_opts(model_opt)
		vocab = torch.load(opt.data + '.vocab.pt')
	fields = vocab

	device_id = 0
	device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() and device_id >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Build a model.

	model = onmt.model_builder.build_model(model_opt, opt, fields, checkpoint=None)
	generator = None  # FIXME [implement] >>

	# NOTE [info] >> The generator is not called. So It has to be called explicitly.
	#model.generator = generator
	model.add_module('generator', generator)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# Set up an optimizer.

	lr = 1.0
	torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	optimizer = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, learning_rate_decay_fn=None, max_grad_norm=2)

	#--------------------
	# Train.

	model_saver = onmt.models.build_model_saver(model_opt, opt, model, fields, optimizer)
	#model_saver = None

	trainer = onmt.trainer.build_trainer(opt, device_id, model, fields, optimizer, model_saver=model_saver)

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/translate.py
def translate_test():
	# REF [site] >> https://opennmt.net/OpenNMT-py/options/translate.html
	if True:
		parser = onmt.utils.parse.ArgumentParser(description='translate_test')

		onmt.opts.config_opts(parser)
		onmt.opts.translate_opts(parser)

		opt = parser.parse_args()
	else:
		opt = argparse.Namespace()

		opt.config = None  # Config file path (default: None).
		opt.save_config = None  # Config file save path (default: None).

		# Model.
		opt.model = []  # Path to model .pt file(s). Multiple models can be specified, for ensemble decoding. (default: []).
		opt.fp32 = False # Force the model to be in FP32 because FP16 is very slow on GTX1080(ti). (default: False).
		opt.avg_raw_probs = False  # If this is set, during ensembling scores from different models will be combined by averaging their raw probabilities and then taking the log. Otherwise, the log probabilities will be averaged directly. Necessary for models whose output layers can assign zero probability. (default: False).

		# Data.
		opt.data_type = 'text'  # Type of the source input. Options: [text | img]. (default: text).
		opt.src = None  # Source sequence to decode (one line per sequence) (default: None).
		opt.src_dir = ''  # Source directory for image or audio files (default: ).
		opt.tgt = None  # True target sequence (optional) (default: None).
		opt.shard_size = 10000  # Divide src and tgt (if applicable) into smaller multiple src and tgt files, then build shards, each shard will have opt.shard_size samples except last shard. shard_size=0 means no segmentation shard_size>0 means segment dataset into multiple shards, each shard has shard_size samples (default: 10000).
		opt.output = 'pred.txt'  # Path to output the predictions (each line will be the decoded sequence (default: pred.txt).
		opt.report_align = False  # Report alignment for each translation. (default: False).
		opt.report_time = False  # Report some translation time metrics (default: False).
		opt.dynamic_dict = False  # Create dynamic dictionaries (default: False).
		opt.share_vocab = False  # Share source and target vocabulary (default: False).

		# Random Sampling.
		opt.random_sampling_topk = 1  # Set this to -1 to do random sampling from full distribution. Set this to value k>1 to do random sampling restricted to the k most likely next tokens. Set this to 1 to use argmax or for doing beam search. (default: 1).
		opt.random_sampling_temp = 1.0  # If doing random sampling, divide the logits by this before computing softmax during decoding. (default: 1.0).
		opt.seed = 829  # Random seed (default: 829).

		# Beam.
		opt.beam_size = 5  # Beam size (default: 5).
		opt.min_length = 0  # Minimum prediction length (default: 0).
		opt.max_length = 100  # Maximum prediction length. (default: 100).
		opt.max_sent_length = None  # Deprecated, use '-max_length' instead (default: None).
		opt.stepwise_penalty = False  # Apply penalty at every decoding step. Helpful for summary penalty. (default: False).
		opt.length_penalty = 'none'  # Length Penalty to use. {none, wu, avg} (default: none).
		opt.ratio = -0.0  # Ratio based beam stop condition (default: -0.0).
		opt.coverage_penalty = 'none'  # Coverage Penalty to use. {none, wu, summary} (default: none).
		opt.alpha = 0.0  # Google NMT length penalty parameter (higher = longer generation) (default: 0.0).
		opt.beta = -0.0  # Coverage penalty parameter (default: -0.0).
		opt.block_ngram_repeat = 0  # Block repetition of ngrams during decoding. (default: 0).
		opt.ignore_when_blocking = []  # Ignore these strings when blocking repeats. You want to block sentence delimiters. (default: []).
		opt.replace_unk = False  # Replace the generated UNK tokens with the source token that had highest attention weight. If phrase_table is provided, it will look up the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table), then it will copy the source token. (default: False).
		opt.phrase_table = ''  # If phrase_table is provided (with replace_unk), it will look up the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table), then it will copy the source token. (default: )

		# Logging.
		opt.verbose = False  # Print scores and predictions for each sentence (default: False).
		opt.log_file = ''  # Output logs to a file under this path. (default: ).
		opt.log_file_level = '0'  # {CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET, 50, 40, 30, 20, 10, 0}.
		opt.attn_debug = False  # Print best attn for each word (default: False).
		opt.align_debug = False  # Print best align for each word (default: False).
		opt.dump_beam = ''  # File to dump beam information to. (default: ).
		opt.n_best = 1  # If verbose is set, will output the n_best decoded sentences (default: 1).

		# Efficiency.
		opt.batch_size = 30  # Batch size (default: 30).
		opt.batch_type = 'sents'  # Batch grouping for batch_size. Standard is sents. Tokens will do dynamic batching {sents, tokens} (default: sents).
		opt.gpu = -1  # Device to run on (default: -1).

		# Speech.
		opt.sample_rate = 16000  # Sample rate. (default: 16000).
		opt.window_size = 0.02  # Window size for spectrogram in seconds (default: 0.02).
		opt.window_stride = 0.01  # Window stride for spectrogram in seconds (default: 0.01).
		opt.window = 'hamming'  # Window type for spectrogram generation (default: hamming).

		# Image.
		opt.image_channel_size = 3  # Using grayscale image can training model faster and smaller {3, 1} (default: 3).

	print('Translate options:\n{}'.format(opt))

	#------------------------------------------------------------
	#onmt.bin.translate.translate(opt)

	#------------------------------------------------------------
	# REF [function] >> translate() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/bin/translate.py.
	onmt.utils.parse.ArgumentParser.validate_translate_opts(opt)
	logger = onmt.utils.logging.init_logger(opt.log_file)

	translator = onmt.translate.translator.build_translator(opt, report_score=True, logger=None, out_file=None)

	src_shards = onmt.utils.misc.split_corpus(opt.src, opt.shard_size)
	tgt_shards = onmt.utils.misc.split_corpus(opt.tgt, opt.shard_size)
	shard_pairs = zip(src_shards, tgt_shards)

	for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
		logger.info("Translating shard {}.".format(i))
		translator.translate(
			src=src_shard,
			tgt=tgt_shard,
			src_dir=opt.src_dir,
			batch_size=opt.batch_size,
			batch_type=opt.batch_type,
			attn_debug=opt.attn_debug,
			align_debug=opt.align_debug
		)

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/server.py
def server_test():
	raise NotImplementedError

#--------------------------------------------------------------------

# REF [site] >> https://opennmt.net/OpenNMT-py/Library.html
def library_example():
	is_trained, is_model_loaded = True, False
	preprocessed_data_dir_path = './data'

	if is_trained:
		model_filepath = './onmt_library_model.pt'
	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	gpu = 0 if torch.cuda.is_available() else -1

	#--------------------
	# Prepare data.

	# Load in the vocabulary for the model of interest.
	vocab_fields = torch.load(preprocessed_data_dir_path + '/data.vocab.pt')
	train_data_files = [
		preprocessed_data_dir_path + '/data.train.0.pt'
	]
	valid_data_files = [
		preprocessed_data_dir_path + '/data.valid.0.pt'
	]

	src_text_field = vocab_fields['src'].base_field
	src_vocab = src_text_field.vocab
	src_padding = src_vocab.stoi[src_text_field.pad_token]

	tgt_text_field = vocab_fields['tgt'].base_field
	tgt_vocab = tgt_text_field.vocab
	tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

	train_iter = onmt.inputters.inputter.DatasetLazyIter(
		dataset_paths=train_data_files, fields=vocab_fields,
		batch_size=50, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
		device=device, is_train=True, repeat=True
	)
	valid_iter = onmt.inputters.inputter.DatasetLazyIter(
		dataset_paths=valid_data_files, fields=vocab_fields,
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

	# Specify the tgt word generator.
	model.generator = torch.nn.Sequential(
		torch.nn.Linear(rnn_size, len(tgt_vocab)),
		torch.nn.LogSoftmax(dim=-1)
	)

	if is_model_loaded:
		model, model.generator = load_model(model_filepath_to_load, model, model.generator, device=device)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# Specify loss computation module.

	loss = onmt.utils.loss.NMTLossCompute(
		criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction='sum'),
		generator=model.generator
	)

	#--------------------
	# Set up an optimizer.

	lr = 1.0
	torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, learning_rate_decay_fn=None, max_grad_norm=2)

	#--------------------
	# Train.

	if is_trained:
		# Keeping track of the output requires a report manager.
		report_manager = onmt.utils.ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
		trainer = onmt.Trainer(
			model=model, train_loss=loss, valid_loss=loss,
			optim=optim, report_manager=report_manager
		)

		print('Start training...')
		start_time = time.time()
		total_stats = trainer.train(
			train_iter=train_iter, train_steps=400,
			valid_iter=valid_iter, valid_steps=200
		)
		print('End training: {} secs.'.format(time.time() - start_time))
		print('Train: Accuracy = {}, Cross entropy = {}, Perplexity = {}.'.format(total_stats.accuracy(), total_stats.xent(), total_stats.ppl()))

		save_model(model_filepath, model, model.generator)

	#--------------------
	# Load up the translation functions.

	src_reader = onmt.inputters.str2reader['text']
	tgt_reader = onmt.inputters.str2reader['text']
	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')
	# Decoding strategy:
	#	Greedy search, if beam_size = 1.
	#	Beam search, otherwise.
	translator = onmt.translate.Translator(
		model=model, fields=vocab_fields, 
		src_reader=src_reader(), tgt_reader=tgt_reader(), 
		global_scorer=scorer, gpu=gpu
	)
	# Build a word-based translation from the batch output of translator and the underlying dictionaries.
	builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_files[0]), fields=vocab_fields)

	for batch in valid_iter:
		print('Start translating...')
		start_time = time.time()
		trans_batch = translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=False)
		print('End translating: {} secs.'.format(time.time() - start_time))

		translations = builder.from_batch(trans_batch)
		for trans in translations:
			print(trans.log(0))

#--------------------------------------------------------------------

def build_im2latex_model(input_channel, num_classes, word_vec_size):
	bidirectional_encoder = False
	embedding_dropout = 0.3
	encoder_num_layers = 2
	encoder_rnn_size = 500
	encoder_dropout = 0.3
	decoder_rnn_type = 'LSTM'
	decoder_num_layers = 2
	decoder_hidden_size = 500
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

	model = onmt.models.NMTModel(encoder, decoder)

	return model, generator

def im2latex_example():
	src_data_type, tgt_data_type = 'img', 'text'
	input_channel = 3
	num_classes = 466
	word_vec_size = 500
	batch_size = 32

	is_trained, is_model_loaded = True, False

	# TODO [choose] >>
	if True:
		# For im2text_small.
		# REF [site] >> http://lstm.seas.harvard.edu/latex/im2text_small.tgz
		preprocessed_data_dir_path = './data/im2text_small'
		num_train_data_files, num_valid_data_files = 2, 1
	else:
		# For im2text.
		# REF [site] >> http://lstm.seas.harvard.edu/latex/im2text.tgz
		preprocessed_data_dir_path = './data/im2text'
		num_train_data_files, num_valid_data_files = 153, 17

	if is_trained:
		model_filepath = './data/im2latex_model.pt'
	if is_model_loaded:
		# Downloaded from http://lstm.seas.harvard.edu/latex/py-model.pt.
		#model_filepath_to_load = './data/py-model.pt'
		model_filepath_to_load = './data/im2latex_model.pt'
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	def read_lines_from_file(filepath):
		try:
			with open(filepath, 'r', encoding='utf-8') as fd:
				lines = fd.read().splitlines()  # A list of strings.
				return lines
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(filepath))
			raise
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(filepath))
			raise

	# REF [site] >> https://opennmt.net/OpenNMT-py/im2text.html

	# TODO [choose] >>
	# NOTE [info] >> Two vocab_fields's are different, so a model has to be trained.
	#	If not, wrong results will be obtained.
	if True:
		# NOTE [info] >> When preprocessing data by onmt_preprocess or ${OpenNMT-py_HOME}/onmt/bin/preprocess.py.

		# Load in the vocabulary for the model of interest.
		vocab_fields = torch.load(preprocessed_data_dir_path + '/demo.vocab.pt')
	else:
		#UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = '<UNK>', '<PAD>', '<SOS>', '<EOS>'
		UNKNOWN_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = '<unk>', '<blank>', '<s>', '</s>'

		def preprocess(x):
			return x
		def postprocess(batch, vocab):
			if len(batch) == 1: return batch[0].unsqueeze(dim=0)
			max_height, max_width = max([tt.shape[1] for tt in batch]), max([tt.shape[2] for tt in batch])
			batch_resized = torch.zeros((len(batch), 3, max_height, max_width), dtype=batch[0].dtype)
			for idx, tt in enumerate(batch):
				batch_resized[idx, :, :tt.shape[1], :tt.shape[2]] = tt
			return batch_resized

		# REF [function] >> image_fields() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/inputters/image_dataset.py.
		src_field = torchtext.data.Field(
			sequential=False, use_vocab=False, init_token=None, eos_token=None, fix_length=None,
			#dtype=torch.float32, preprocessing=preprocess, postprocessing=postprocess, lower=False,
			dtype=torch.float32, preprocessing=None, postprocessing=postprocess, lower=False,
			tokenize=None, tokenizer_language='en',
			include_lengths=False, batch_first=False, pad_token=None, pad_first=False, unk_token=UNKNOWN_TOKEN,
			truncate_first=False, stop_words=None, is_target=False
		)
		tgt_field = torchtext.data.Field(
			sequential=True, use_vocab=True, init_token=SOS_TOKEN, eos_token=EOS_TOKEN, fix_length=None,
			dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
			tokenize=None, tokenizer_language='en',
			#tokenize=functools.partial(onmt.inputters.inputter._feature_tokenize, layer=0, feat_delim=None, truncate=None), tokenizer_language='en',
			include_lengths=False, batch_first=False, pad_token=PAD_TOKEN, pad_first=False, unk_token=UNKNOWN_TOKEN,
			truncate_first=False, stop_words=None, is_target=False
		)
		indices_field = torchtext.data.Field(
			sequential=False, use_vocab=False, init_token=None, eos_token=None, fix_length=None,
			dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
			tokenize=None, tokenizer_language='en',
			include_lengths=False, batch_first=False, pad_token=None, pad_first=False, unk_token=UNKNOWN_TOKEN,
			truncate_first=False, stop_words=None, is_target=False
		)
		corpus_id_field = torchtext.data.Field(
			sequential=False, use_vocab=True, init_token=None, eos_token=None, fix_length=None,
			dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
			tokenize=None, tokenizer_language='en',
			include_lengths=False, batch_first=False, pad_token=None, pad_first=False, unk_token=UNKNOWN_TOKEN,
			truncate_first=False, stop_words=None, is_target=False
		)

		# NOTE [info] >> It is better to build a vocabulary from corpora.
		if True:
			tgt_train_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-train.txt')
			tgt_valid_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-val.txt')
			tgt_test_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-test.txt')
			texts = [txt.split() for txt in tgt_train_texts] + [txt.split() for txt in tgt_valid_texts] + [txt.split() for txt in tgt_test_texts]
			tgt_field.build_vocab(texts)
		else:
			vocab = read_lines_from_file(preprocessed_data_dir_path + '/vocab.txt')
			#tgt_field.vocab = vocab  # AttributeError: 'list' object has no attribute 'stoi'.
			tgt_field.build_vocab([vocab])
		corpus_id_field.build_vocab(['train'])

		# REF [function] >> build_vocab() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/inputters/inputter.py.
		vocab_fields = {
			'src': src_field,
			'tgt': onmt.inputters.text_dataset.TextMultiField('tgt', tgt_field, feats_fields=[]),
			'indices': indices_field,
			'corpus_id': corpus_id_field,
		}

	#src_text_field = vocab_fields['src'].base_field  # Error: AttributeError: 'Field' object has no attribute 'base_field'.
	#src_vocab = src_text_field.vocab
	#src_padding = src_vocab.stoi[src_text_field.pad_token]

	tgt_text_field = vocab_fields['tgt'].base_field
	tgt_vocab = tgt_text_field.vocab
	tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

	src_reader = onmt.inputters.str2reader[src_data_type]
	tgt_reader = onmt.inputters.str2reader[tgt_data_type]
	if src_data_type == 'img':
		src_reader_obj = src_reader(truncate=None, channel_size=input_channel)
	elif src_data_type == 'audio':
		src_reader_obj = src_reader(sample_rate=0, window_size=0, window_stride=0, window=None, normalize_audio=True, truncate=None)
	else:
		src_reader_obj = src_reader()
	if tgt_data_type == 'img':
		tgt_reader_obj = tgt_reader(truncate=None, channel_size=input_channel)
	elif tgt_data_type == 'audio':
		tgt_reader_obj = tgt_reader(sample_rate=0, window_size=0, window_stride=0, window=None, normalize_audio=True, truncate=None)
	else:
		tgt_reader_obj = tgt_reader()

	# TODO [choose] >>
	if True:
		# NOTE [info] >> When preprocessing data by onmt_preprocess or ${OpenNMT-py_HOME}/onmt/bin/preprocess.py.

		train_data_files = list()
		for idx in range(num_train_data_files):
			train_data_files.append(preprocessed_data_dir_path + '/demo.train.{}.pt'.format(idx))
		valid_data_files = list()
		for idx in range(num_valid_data_files):
			valid_data_files.append(preprocessed_data_dir_path + '/demo.valid.{}.pt'.format(idx))

		# REF [function] >> build_dataset_iter() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/inputters/inputter.py.
		train_iter = onmt.inputters.inputter.DatasetLazyIter(
			dataset_paths=train_data_files, fields=vocab_fields,
			batch_size=batch_size, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
			device=device, is_train=True, repeat=True,
			num_batches_multiple=1, yield_raw_example=False
		)
		valid_iter = onmt.inputters.inputter.DatasetLazyIter(
			dataset_paths=valid_data_files, fields=vocab_fields,
			batch_size=batch_size, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
			device=device, is_train=False, repeat=False,
			num_batches_multiple=1, yield_raw_example=False
		)
	else:
		sortkey = onmt.inputters.str2sortkey[tgt_data_type]

		src_dir_path = preprocessed_data_dir_path + '/images'

		src_train_filepaths = read_lines_from_file(preprocessed_data_dir_path + '/src-train.txt')
		src_train_filepaths = [bytes(fpath, encoding='utf-8') for fpath in src_train_filepaths]
		tgt_train_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-train.txt')
		src_valid_filepaths = read_lines_from_file(preprocessed_data_dir_path + '/src-val.txt')
		src_valid_filepaths = [bytes(fpath, encoding='utf-8') for fpath in src_valid_filepaths]
		tgt_valid_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-val.txt')

		# REF [function] >> translate() in https://github.com/OpenNMT/OpenNMT-py/tree/master/onmt/translate/translator.py.
		train_src_data = {'reader': src_reader_obj, 'data': src_train_filepaths, 'dir': src_dir_path}
		train_tgt_data = {'reader': tgt_reader_obj, 'data': tgt_train_texts, 'dir': None}
		train_readers, train_data, train_dirs = onmt.inputters.Dataset.config([('src', train_src_data), ('tgt', train_tgt_data)])
		train_dataset = onmt.inputters.Dataset(
			fields=vocab_fields, readers=train_readers, data=train_data, dirs=train_dirs, sort_key=sortkey,
			filter_pred=None, corpus_id=None
		)
		valid_src_data = {'reader': src_reader_obj, 'data': src_valid_filepaths, 'dir': src_dir_path}
		valid_tgt_data = {'reader': tgt_reader_obj, 'data': tgt_valid_texts, 'dir': None}
		valid_readers, valid_data, valid_dirs = onmt.inputters.Dataset.config([('src', valid_src_data), ('tgt', valid_tgt_data)])
		valid_dataset = onmt.inputters.Dataset(
			fields=vocab_fields, readers=valid_readers, data=valid_data, dirs=valid_dirs, sort_key=sortkey,
			filter_pred=None, corpus_id=None
		)

		train_iter = onmt.inputters.inputter.OrderedIterator(
			dataset=train_dataset,
			batch_size=batch_size, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
			device=device, train=True, repeat=True,
			sort=False, sort_within_batch=True,
			yield_raw_example=False
		)
		#train_iter.create_batches()
		valid_iter = onmt.inputters.inputter.OrderedIterator(
			dataset=valid_dataset,
			batch_size=batch_size, batch_size_multiple=1, batch_size_fn=None, pool_factor=8192,
			device=device, train=False, repeat=False,
			sort=False, sort_within_batch=True,
			yield_raw_example=False
		)
		#valid_iter.create_batches()

	#--------------------
	# Build a model.

	model, generator = build_im2latex_model(input_channel, num_classes, word_vec_size)
	#if model: print('Model:\n{}'.format(model))

	# NOTE [info] >> The generator is not called. So It has to be called explicitly.
	#model.generator = generator
	model.add_module('generator', generator)

	if is_model_loaded:
		model, generator = load_model(model_filepath_to_load, model, generator, device=device)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# Specify loss computation module.

	loss = onmt.utils.loss.NMTLossCompute(
		criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction='sum'),
		generator=model.generator
	)

	#--------------------
	# Set up an optimizer.

	lr = 1.0
	torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, learning_rate_decay_fn=None, max_grad_norm=2)

	#--------------------
	# Train.

	if is_trained:
		# Keeping track of the output requires a report manager.
		report_manager = onmt.utils.ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
		trainer = onmt.Trainer(
			model=model, train_loss=loss, valid_loss=loss,
			optim=optim, report_manager=report_manager
		)

		print('Start training...')
		start_time = time.time()
		total_stats = trainer.train(
			train_iter=train_iter, train_steps=400,
			save_checkpoint_steps=5000,
			valid_iter=valid_iter, valid_steps=200
		)
		print('End training: {} secs.'.format(time.time() - start_time))
		print('Train: Accuracy = {}, Cross entropy = {}, Perplexity = {}.'.format(total_stats.accuracy(), total_stats.xent(), total_stats.ppl()))

		print('Start evaluating...')
		start_time = time.time()
		stats = trainer.validate(valid_iter=valid_iter, moving_average=None)
		print('End evaluating: {} secs.'.format(time.time() - start_time))
		print('Evaluation: Accuracy = {}, Cross entropy = {}, Perplexity = {}.'.format(stats.accuracy(), stats.xent(), stats.ppl()))

		save_model(model_filepath, model, model.generator)

	#--------------------
	# Load up the translation functions.

	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')

	# TODO [choose] >>
	if True:
		# NOTE [info] >> When using image files.

		try:
			import tempfile
			with tempfile.TemporaryFile(mode='w') as fd:
				# Decoding strategy:
				#	Greedy search, if beam_size = 1.
				#	Beam search, otherwise.
				translator = onmt.translate.Translator(
					model=model, fields=vocab_fields,
					src_reader=src_reader_obj, tgt_reader=tgt_reader_obj,
					n_best=1, min_length=0, max_length=100,
					beam_size=30, random_sampling_topk=1, random_sampling_temp=1,
					data_type=src_data_type,
					global_scorer=scorer,
					copy_attn=False, report_align=False, report_score=True, out_file=fd,
					gpu=gpu
				)

				src_filepaths = read_lines_from_file(preprocessed_data_dir_path + '/src-test.txt')
				src_filepaths = [bytes(fpath, encoding='utf-8') for fpath in src_filepaths]
				tgt_texts = read_lines_from_file(preprocessed_data_dir_path + '/tgt-test.txt')
				try:
					print('Start translating...')
					start_time = time.time()
					scores, predictions = translator.translate(src=src_filepaths, tgt=None, src_dir=preprocessed_data_dir_path + '/images', batch_size=batch_size, batch_type='tokens', attn_debug=False, align_debug=False, phrase_table='')
					#scores, predictions = translator.translate(src=src_filepaths, tgt=tgt_texts, src_dir=preprocessed_data_dir_path + '/images', batch_size=batch_size, batch_type='tokens', attn_debug=False, align_debug=False, phrase_table='')
					print('End translating: {} secs.'.format(time.time() - start_time))

					for idx, (score, pred, gt) in enumerate(zip(scores, predictions, tgt_texts)):
						print('ID #{}:'.format(idx))
						print('\tG/T        = {}.'.format(gt))
						print('\tPrediction = {}.'.format(pred[0]))
						print('\tScore      = {}.'.format(score[0].cpu().item()))
				except (RuntimeError, Exception) as ex:
					print("Error: {}.".format(str(ex)))
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(ex))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(ex))
	else:
		# Decoding strategy:
		#	Greedy search, if beam_size = 1.
		#	Beam search, otherwise.
		translator = onmt.translate.Translator(
			model=model, fields=vocab_fields,
			src_reader=src_reader_obj, tgt_reader=tgt_reader_obj,
			n_best=1, min_length=0, max_length=100,
			beam_size=30, random_sampling_topk=1, random_sampling_temp=1,
			data_type=src_data_type,
			global_scorer=scorer,
			copy_attn=False, report_align=False, report_score=True,
			gpu=gpu
		)

		for batch in valid_iter:
			print('Start translating...')
			start_time = time.time()
			trans_batch = translator.translate_batch(batch=batch, src_vocabs=[], attn_debug=False)
			print('End translating: {} secs.'.format(time.time() - start_time))

			#print('\tBatch source = {}.'.format(trans_batch['batch'].src.cpu().numpy()))
			#print('\tBatch target = {}.'.format(trans_batch['batch'].tgt.cpu().numpy()))
			#print('\tBatch indices = {}.'.format(trans_batch['batch'].indices.cpu().numpy()))
			#print('\tBatch corpus ID = {}.'.format(trans_batch['batch'].corpus_id.cpu().numpy()))

			for idx, (pred, score, attn, gold_score, alignment) in enumerate(zip(trans_batch['predictions'], trans_batch['scores'], trans_batch['attention'], trans_batch['gold_score'], trans_batch['alignment'])):
				print('ID #{}:'.format(idx))
				try:
					print('\tPrediction = {}.'.format(' '.join([tgt_vocab.itos[elem] for elem in pred[0].cpu().numpy() if elem < len(tgt_vocab.itos)])))
				except IndexError as ex:
					print('\tDecoding error: {}.'.format(pred[0]))
				print('\tScore      = {}.'.format(score[0].cpu().item()))
				#print('\tAttention  = {}.'.format(attn[0].cpu().numpy()))
				print('\tGold score = {}.'.format(gold_score.cpu().numpy()))
				#print('\tAlignment  = {}.'.format(alignment[0].cpu().item()))

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
	def __init__(self, encoder, decoder, generator=None):
		super().__init__()

		self.encoder, self.decoder, self._generator = encoder, decoder, generator

	# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
	def forward(self, src, tgt, lengths, bptt=False, with_align=False):
		# TODO [check] >> This function is not tested.
		dec_in = tgt[:-1]  # Exclude last target from inputs.
		enc_state, memory_bank, lengths = self.encoder(src, lengths=lengths)
		if bptt is False:
			self.decoder.init_state(src, memory_bank, enc_state)
		dec_outs, attns = self.decoder(dec_in, memory_bank, memory_lengths=lengths, with_align=with_align)
		if self._generator: dec_outs = self._generator(dec_outs)
		return dec_outs, attns

# REF [site] >> https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/model_builder.py
def build_my_im2txt_model(use_NMTModel, input_channel, num_classes, word_vec_size):
	encoder, decoder, generator = build_submodels(input_channel, num_classes, word_vec_size)

	if use_NMTModel:
		model = onmt.models.NMTModel(encoder, decoder)
	else:
		model = MyModel(encoder, decoder, generator=None)

	return model, generator

def simple_example():
	use_NMTModel = False
	input_channel = 3
	num_classes = 1798
	word_vec_size = 80
	batch_size = 64
	max_time_steps = 10

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Build a model.

	model, generator = build_my_im2txt_model(use_NMTModel, input_channel, num_classes, word_vec_size)
	#if model: print('Model:\n{}'.format(model))

	# NOTE [info] >> The generator is not called. So It has to be called explicitly.
	#model.generator = generator
	model.add_module('generator', generator)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# For checking.

	if False:
		inputs = torch.randn(batch_size, input_channel, 300, 300)
		outputs = torch.randint(num_classes, (batch_size, max_time_steps, 1))
		outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].
		output_lens = torch.randint(1, max_time_steps + 1, (batch_size,))

		inputs = inputs.to(device)
		outputs, output_lens = outputs.to(device), output_lens.to(device)

		with torch.no_grad():
			model_outputs, attentions = model(inputs, outputs, output_lens)  # [target length, batch size, hidden size] & [target length, batch size, source length].
			model_outputs = model.generator(model_outputs)

		#model_outputs = model_outputs.transpose(0, 1)  # [T, B, F] -> [B, T, F].
		model_outputs = model_outputs.cpu().numpy()
		attentions = attentions['std'].cpu().numpy()

		print("Model outputs' shape =", model_outputs.shape)
		print("Attentions' shape =", attentions.shape)

def main():
	#preprocess_test()  # Not yet completed.
	#train_test()  # Not yet completed.
	#translate_test()  # Not yet completed.
	#server_test()  # Not yet implemented.

	#--------------------
	#library_example()

	im2latex_example()
	#simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
