#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, json
import numpy as np
import tensorflow as tf
import model, sample, encoder

# REF [site] >> https://www.analyticsvidhya.com/blog/2019/07/openai-gpt2-text-generator-python/
def simple_gpt2_example():
	# REF [site] >> https://github.com/openai/gpt-2

	if 'posix' == os.name:
		lib_base_dir_path = '/home/sangwook/lib_repo/python'
	else:
		lib_base_dir_path = 'D:/lib_repo/python/rnd'
	lib_dir_path = lib_base_dir_path + '/gpt-2_github'

	import sys
	sys.path.append(lib_dir_path + '/src')

	#-----
	models_dir = lib_dir_path + '/models'
	model_name = '345M'
	seed = None
	nsamples = 1
	batch_size = 1
	length = 300
	temperature = 1
	top_k = 0

	raw_text = 'I went to a lounge to celebrate my birthday and'

	models_dir = os.path.expanduser(os.path.expandvars(models_dir))
	if batch_size is None:
		batch_size = 1
	assert nsamples % batch_size == 0

	enc = encoder.get_encoder(model_name, models_dir)
	hparams = model.default_hparams()
	with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
		hparams.override_from_dict(json.load(f))

	if length is None:
		length = hparams.n_ctx // 2
	elif length > hparams.n_ctx:
		raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

	with tf.Session(graph=tf.Graph()) as sess:
		context = tf.placeholder(tf.int32, [batch_size, None])
		np.random.seed(seed)
		tf.set_random_seed(seed)
		output = sample.sample_sequence(
			hparams=hparams, length=length,
			context=context,
			batch_size=batch_size,
			temperature=temperature, top_k=top_k
		)

		saver = tf.train.Saver()
		ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
		saver.restore(sess, ckpt)

		#--------------------
		context_tokens = enc.encode(raw_text)
		generated = 0
		for _ in range(nsamples // batch_size):
			out = sess.run(output, feed_dict={
				context: [context_tokens for _ in range(batch_size)]
			})[:, len(context_tokens):]
			for i in range(batch_size):
				generated += 1
				text = enc.decode(out[i])
				print('=' * 40 + ' SAMPLE ' + str(generated) + ' ' + '=' * 40)
				print(text)
		print('=' * 80)

def main():
	simple_gpt2_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
