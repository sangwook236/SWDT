import numpy as np
import yaml
import caffe

class MyLayer(caffe.Layer):
	def setup(self, bottom, top):
		self.num = yaml.load(self.param_str)['num']
		print('Parameter num :', self.num)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		top[0].reshape(*bottom[0].shape)
		top[0].data[...] = bottom[0].data + self.num

	def backward(self, top, propagate_down, bottom):
		pass
