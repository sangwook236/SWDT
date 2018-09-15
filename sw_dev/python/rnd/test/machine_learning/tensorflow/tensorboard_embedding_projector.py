# REF [site] >> https://www.tensorflow.org/get_started/embedding_viz
# REF [site] >> http://stackoverflow.com/questions/41258391/tensorboard-embedding-example
# REF [site] >> http://stackoverflow.com/questions/41258391/tensorboard-embedding-example/41262360#41262360

import os
import tensorflow as tf
import numpy as np
import fasttext
from tensorflow.contrib.tensorboard.plugins import projector

# Load model.
word2vec = fasttext.load_model('wiki.en.bin')

# Create a list of vectors.
embedding = np.empty((len(word2vec.words), word2vec.dim), dtype=np.float32)
for i, word in enumerate(word2vec.words):
    embedding[i] = word2vec[word]

# Write labels.
with open('log/metadata.tsv', 'w') as f:
    for word in word2vec.words:
        f.write(word + '\n')

# Setup a TensorFlow session.
tf.reset_default_graph()
sess = tf.InteractiveSession()

embedding_var = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(embedding_var, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# Create a TensorFlow summary writer.
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()

embedding_conf = config.embeddings.add()
#embedding_conf.tensor_name = embedding_var.name
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')

projector.visualize_embeddings(summary_writer, config)

# Save the model.
saver = tf.train.Saver()
saver.save(sess, os.path.join('log', 'tf_ckpt'))

# TensorFlow graph to TensorBoard log
#	REF [file] >> tensorflow_saving_and_loading.py

# Usage:
#	tensorboard --logdir=log
