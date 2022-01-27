#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import torch, torchvision
import matplotlib.pyplot as plt

def visualize_learning_rate_decay_policy(scheduler, max_step=100, ylim=None, title=None):
	lrs = list()
	for _ in range(max_step):
		scheduler.step()
		lrs.append(scheduler.get_last_lr())

	plt.figure()
	plt.plot(range(max_step), lrs)
	plt.title(title if title else "Learning rate decay policy")
	plt.xlabel("Step")
	plt.ylabel("Learning rate")
	if ylim: plt.ylim(ylim)
	plt.grid(visible=True)
	plt.tight_layout()
	plt.show()

def learning_rate_decay_policy_test():
	model = torchvision.models.resnet18()

	init_lr = 1.0
	optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
	#optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

	# LambdaLR.
	if True:
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.95**step)
		visualize_learning_rate_decay_policy(scheduler, max_step=100, ylim=[0, 1], title="LambdaLR")

		def my_lr_func(epoch):
			if epoch < 40:
				return 0.5
			elif epoch < 70:
				return 0.5**2
			elif epoch < 90:
				return 0.5**3
			else:
				return 0.5**4

		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=my_lr_func)
		visualize_learning_rate_decay_policy(scheduler, max_step=100, ylim=[0, 1], title="LambdaLR")

		# REF [site] >> https://gaussian37.github.io/dl-pytorch-lr_scheduler/
		class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
			""" Linear warmup and then constant.
				Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
				Keeps learning rate schedule equal to 1. after warmup_steps.
			"""
			def __init__(self, optimizer, warmup_steps, last_epoch=-1):

				def _lr_func(step):
					if step < warmup_steps:
						return float(step) / float(max(1.0, warmup_steps))
					return 1.

				super(WarmupConstantSchedule, self).__init__(optimizer, _lr_func, last_epoch=last_epoch)

		scheduler = WarmupConstantSchedule(optimizer, warmup_steps=10)
		visualize_learning_rate_decay_policy(scheduler, max_step=100, title="LambdaLR")

	# StepLR.
	if False:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="StepLR")

	# MultiStepLR.
	if False:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 350], gamma=0.5)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="MultiStepLR")

	# ExponentialLR.
	if False:
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
		visualize_learning_rate_decay_policy(scheduler, max_step=100, ylim=[0, 1], title="ExponentialLR")

	# CyclicLR.
	if False:
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.9, step_size_up=50, step_size_down=100, mode="triangular")
		visualize_learning_rate_decay_policy(scheduler, max_step=500, ylim=[0, 1], title="CyclicLR")

		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.9, step_size_up=50, step_size_down=None, mode="triangular2")
		visualize_learning_rate_decay_policy(scheduler, max_step=500, ylim=[0, 1], title="CyclicLR")

		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.9, step_size_up=50, step_size_down=None, mode="exp_range", gamma=0.995)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, ylim=[0, 1], title="CyclicLR")

	# CosineAnnealingLR.
	if False:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.1)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="CosineAnnealingLR")

	# CosineAnnealingWarmRestarts.
	if False:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="CosineAnnealingWarmRestarts")

# REF [site] >> https://gaussian37.github.io/dl-pytorch-lr_scheduler/
class CosineAnnealingWarmUpRestarts(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1):
		if T_0 <= 0 or not isinstance(T_0, int):
			raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
		if T_mult < 1 or not isinstance(T_mult, int):
			raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
		if T_up < 0 or not isinstance(T_up, int):
			raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
		self.T_0 = T_0
		self.T_mult = T_mult
		self.base_eta_max = eta_max
		self.eta_max = eta_max
		self.T_up = T_up
		self.T_i = T_0
		self.gamma = gamma
		self.cycle = 0
		self.T_cur = last_epoch
		super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
	
	def get_last_lr(self):
		if self.T_cur == -1:
			return self.base_lrs
		elif self.T_cur < self.T_up:
			return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
		else:
			return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
					for base_lr in self.base_lrs]

	def get_lr(self):
		return self.get_last_lr()

	def step(self, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
			self.T_cur = self.T_cur + 1
			if self.T_cur >= self.T_i:
				self.cycle += 1
				self.T_cur = self.T_cur - self.T_i
				self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
		else:
			if epoch >= self.T_0:
				if self.T_mult == 1:
					self.T_cur = epoch % self.T_0
					self.cycle = epoch // self.T_0
				else:
					n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
					self.cycle = n
					self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
					self.T_i = self.T_0 * self.T_mult ** (n)
			else:
				self.T_i = self.T_0
				self.T_cur = epoch
				
		self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
		self.last_epoch = math.floor(epoch)
		for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
			param_group['lr'] = lr

# Noam learning rate decay policy.
#	Step-based, but not epoch-based, learning rate decay policy.
#	REF [paper] >> "Attention Is All You Need", NIPS 2017.
#	REF [site] >> https://nlp.seas.harvard.edu/2018/04/03/attention.html
class NoamLR(object):
	def __init__(self, optimizer, dim_feature, warmup_steps, factor=1, last_step=0):
		self.optimizer = optimizer
		self.dim_feature = dim_feature
		self.warmup_steps = warmup_steps
		self.factor = factor
		self.last_step = last_step
		self.learning_rate = 0

		self.step()  # TODO [check] >> Do we need to call this?

		"""
		# Initialize step and base learning rates.
		if last_step == -1:
			for group in optimizer.param_groups:
				group.setdefault('initial_lr', group['lr'])
		else:
			for i, group in enumerate(optimizer.param_groups):
				if 'initial_lr' not in group:
					raise KeyError("param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))
		self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
		"""

	def get_last_lr(self):
		return self.learning_rate

	def get_lr(self):
		return self.get_last_lr()

	def step(self, step=None):
		if step is None:
			self.last_step += 1
		else:
			self.last_step = step
		self.learning_rate = self._adjust_learning_rate(self.optimizer, self.last_step, self.warmup_steps, self.dim_feature, self.factor)

	@staticmethod
	def _adjust_learning_rate(optimizer, step, warmup_steps, dim_feature, factor):
		lr = factor * (dim_feature**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5)))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return lr

def custom_learning_rate_decay_policy_test():
	model = torchvision.models.resnet18()

	if True:
		init_lr, max_lr = 0.0, 1.0
		optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

		scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=max_lr, T_up=10, gamma=0.5)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="CosineAnnealingWarmUpRestarts")

		scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=max_lr, T_up=10, gamma=0.5)
		visualize_learning_rate_decay_policy(scheduler, max_step=500, title="CosineAnnealingWarmUpRestarts")

		max_lr = 0.5
		num_epochs = 50
		scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=num_epochs // 2, T_mult=1, eta_max=max_lr, T_up=num_epochs // 10, gamma=0.5)
		visualize_learning_rate_decay_policy(scheduler, max_step=num_epochs, title="CosineAnnealingWarmUpRestarts")

	if True:
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=False)

		factor = 1.0
		scheduler = NoamLR(optimizer, dim_feature=256, warmup_steps=2000, factor=factor)  # Step-based, not epoch-based, learning rate policy.
		visualize_learning_rate_decay_policy(scheduler, max_step=20000, title="NoamLR")

		scheduler = NoamLR(optimizer, dim_feature=512, warmup_steps=2000, factor=factor)  # Step-based, not epoch-based, learning rate policy.
		visualize_learning_rate_decay_policy(scheduler, max_step=20000, title="NoamLR")

		factor = 5.0
		scheduler = NoamLR(optimizer, dim_feature=512, warmup_steps=2000, factor=factor)  # Step-based, not epoch-based, learning rate policy.
		visualize_learning_rate_decay_policy(scheduler, max_step=20000, title="NoamLR")

def main():
	#learning_rate_decay_policy_test()
	custom_learning_rate_decay_policy_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
