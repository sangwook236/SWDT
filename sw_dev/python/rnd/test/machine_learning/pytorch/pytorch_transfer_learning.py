#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, typing, copy, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

plt.ion()  # Interactive mode.

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # Pause a bit so that plots are updated.

def train_model(model, dataloaders, device, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase.
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode.
			else:
				model.eval()  # Set model to evaluate mode.

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:  # NOTE [info] >> Errors incluing "photoshop" occurred in pillow 6.0.0.
				inputs = inputs.to(device)
				labels = labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward.
				# Track history if only in train.
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# Backward + optimize only if in training phase.
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# Statistics.
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# Deep copy the model.
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# Load best model weights.
	model.load_state_dict(best_model_wts)
	return model

def visualize_model(model, device, dataloaders, class_names, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloaders['val']):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images // 2, 2, images_so_far)
				ax.axis('off')
				ax.set_title('predicted: {}'.format(class_names[preds[j]]))
				imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return
		model.train(mode=was_training)

# REF [site] >> https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def finetuning_example():
	# Load Data.

	# Data augmentation and normalization for training.
	# Just normalization for validation.
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	data_dir = 'data/hymenoptera_data'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Visualize a few images.
	if False:
		# Get a batch of training data.
		inputs, classes = next(iter(dataloaders['train']))

		# Make a grid from batch.
		out = torchvision.utils.make_grid(inputs)

		imshow(out, title=[class_names[x] for x in classes])

	#--------------------
	# Finetune the convnet.

	# Show a model architecture.
	#print(torchvision.models.resnet18(pretrained=True))

	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 2)

	model_ft = model_ft.to(device)

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized.
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs.
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	# Train and evaluate.
	model_ft = train_model(model_ft, dataloaders, device, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

	visualize_model(model_ft, device, dataloaders, class_names)

# REF [site] >> https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def convnet_as_fixed_feature_extractor_example():
	# Load Data.

	# Data augmentation and normalization for training.
	# Just normalization for validation.
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	data_dir = 'data/hymenoptera_data'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Visualize a few images.
	if False:
		# Get a batch of training data.
		inputs, classes = next(iter(dataloaders['train']))

		# Make a grid from batch.
		out = torchvision.utils.make_grid(inputs)

		imshow(out, title=[class_names[x] for x in classes])

	#--------------------
	model_conv = torchvision.models.resnet18(pretrained=True)

	# Freeze weights.
	for param in model_conv.parameters():
		param.requires_grad = False

	# Parameters of newly constructed modules have requires_grad=True by default.
	num_ftrs = model_conv.fc.in_features
	model_conv.fc = nn.Linear(num_ftrs, 2)

	model_conv = model_conv.to(device)

	criterion = nn.CrossEntropyLoss()

	# Observe that only parameters of final layer are being optimized as opposed to before.
	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs.
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

	#--------------------
	# Train and evaluate.

	model_conv = train_model(model_conv, dataloaders, device, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

	visualize_model(model_conv, device, dataloaders, class_names)

	plt.ioff()
	plt.show()

def resnet_dog_cat_fine_tuning_test():
	def save_model(model_filepath: str, model: torch.nn.Module):
		#torch.save(model.state_dict(), model_filepath)
		torch.save({"state_dict": model.state_dict()}, model_filepath)
		print(f"Saved a model to {model_filepath}.")

	def load_model(model_filepath: str, model: torch.nn.Module, device: torch.device = "cuda"):
		loaded_data = torch.load(model_filepath, map_location=device)
		#model.load_state_dict(loaded_data)
		model.load_state_dict(loaded_data["state_dict"])
		print(f"Loaded a model from {model_filepath}.")
		return model

	timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

	output_dir_path = f"./resnet_dog_cat_fine_tuning_output_{timestamp}"
	if not os.path.exists(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	#--------------------
	# Prepare data.

	class DogCatDataset(torch.utils.data.Dataset):
		def __init__(self, image_dir_path: str, transform: typing.Optional[typing.Callable] = None, target_transform: typing.Optional[typing.Callable] = None):
			self.data = torchvision.datasets.ImageFolder(image_dir_path, transform, target_transform)

			assert len(self.data) > 0, f"No images found in the given directory, {image_dir_path}."

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			s, t = self.data[idx]
			return self.data[idx]

	num_classes = 2  # {'dog', 'cat'}.
	batch_size = 64
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 224)),
		torchvision.transforms.ToTensor(),
	])
	#target_transform = torch.IntTensor
	target_transform = None
	# REF [site] >> https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
	train_dataset = DogCatDataset("./cats_and_dogs_filtered/train", transform=transform, target_transform=target_transform)
	val_dataset = DogCatDataset("./cats_and_dogs_filtered/validation", transform=transform, target_transform=target_transform)
	print(f"#train examples = {len(train_dataset)}, #validation examples = {len(val_dataset)}.")

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, persistent_workers=False)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, persistent_workers=False)
	print(f"#train steps per epoch = {len(train_dataloader)}, #validation steps per epoch = {len(val_dataloader)}.")

	#--------------------
	# Training.
	from torch.utils.tensorboard import SummaryWriter

	def shared_step(model: torch.nn.Module, srcs: torch.Tensor, tgts: torch.Tensor, criterion: torch.nn.Module) -> typing.Dict[str, typing.Any]:
		model_outputs = model(srcs)

		loss = criterion(model_outputs, tgts)

		# Evaluate performance measures.
		#acc = (torch.argmax(model_outputs, dim=-1) == tgts).sum().item()
		acc = (torch.argmax(model_outputs, dim=-1) == tgts).float().mean().item()

		return {
			"loss": loss,
			"acc": acc,
		}
	def training_step(model: torch.nn.Module, batch: typing.Any, batch_idx: typing.Any, criterion: torch.nn.Module, device: torch.device) -> typing.Dict[str, typing.Any]:
		start_time = time.time()
		srcs, tgts = batch
		srcs, tgts = srcs.to(device), tgts.to(device)
		retval = shared_step(model, srcs, tgts, criterion)
		step_time = time.time() - start_time

		return {
			"loss": retval["loss"],
			"acc": retval["acc"],
			"time": step_time,
		}
	validation_step = training_step
	test_step = training_step

	def train_epoch(epoch: int, start_global_step: int, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, is_epoch_based_scheduler: bool, dataloader: torch.utils.data.DataLoader, max_gradient_clipping_norm: typing.Optional[float], device: torch.device, writer: typing.Optional[SummaryWriter] = None) -> typing.Dict[str, typing.Any]:
		model.train()

		global_step = start_global_step
		loss_epoch, acc_epoch, num_examples_epoch = 0.0, 0.0, 0
		for batch_idx, batch in enumerate(dataloader):
			optimizer.zero_grad()

			#model_output = model.training_step(batch, batch_idx, criterion, device)
			model_output = training_step(model, batch, batch_idx, criterion, device)
			loss = model_output["loss"]
			loss.backward()

			if max_gradient_clipping_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clipping_norm)
			optimizer.step()
			if scheduler and not is_epoch_based_scheduler: scheduler.step()  # For step-based scheduler.

			num_examples = len(batch[0])
			loss = loss.item()
			acc = model_output["acc"]
			loss_epoch += loss * num_examples
			acc_epoch += acc * num_examples
			num_examples_epoch += num_examples

			if writer:
				writer.add_scalar("epoch", epoch, global_step)
				if scheduler:
					for idx, lr in enumerate(scheduler.get_last_lr()):
						writer.add_scalar(f"learning_rate[{idx}]", lr, global_step)
				writer.add_scalar("train_loss_step", loss, global_step)
				writer.add_scalar("train_acc_step", acc, global_step)

			global_step += 1
		if scheduler and is_epoch_based_scheduler: scheduler.step()  # For epoch-based scheduler.

		#assert global_step == (epoch + 1) * len(dataloader) - 1, f"{global_step}, {(epoch + 1) * len(dataloader) - 1}"
		if writer:
			writer.add_scalar("train_loss_epoch", loss_epoch / num_examples_epoch, global_step - 1)
			writer.add_scalar("train_acc_epoch", acc_epoch / num_examples_epoch, global_step - 1)

		return {
			"loss": loss_epoch / num_examples_epoch,
			"acc": acc_epoch / num_examples_epoch,
			"global_step": global_step,
		}

	def evaluate_model(global_step: int, model: torch.nn.Module, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, writer: typing.Optional[SummaryWriter] = None) -> typing.Dict[str, typing.Any]:
		model.eval()

		loss_epoch, acc_epoch, num_examples_epoch = 0.0, 0.0, 0
		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader):
				#model_output = model.validation_step(batch, batch_idx, criterion, device)
				model_output = validation_step(model, batch, batch_idx, criterion, device)
				loss = model_output["loss"]

				num_examples = len(batch[0])
				loss = loss.item()
				acc = model_output["acc"]
				loss_epoch += loss * num_examples
				acc_epoch += acc * num_examples
				num_examples_epoch += num_examples

			if writer:
				writer.add_scalar("val_loss", loss_epoch / num_examples_epoch, global_step)
				writer.add_scalar("val_acc", acc_epoch / num_examples_epoch, global_step)

		return {
			"loss": loss_epoch / num_examples_epoch,
			"acc": acc_epoch / num_examples_epoch,
		}

	def test_model(global_step: int, model: torch.nn.Module, criterion: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, writer: typing.Optional[SummaryWriter] = None) -> typing.Dict[str, typing.Any]:
		model.eval()

		loss_epoch, acc_epoch, num_examples_epoch = 0.0, 0.0, 0
		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader):
				#model_output = model.test_step(batch, batch_idx, criterion, device)
				model_output = test_step(model, batch, batch_idx, criterion, device)
				loss = model_output["loss"]

				num_examples = len(batch[0])
				loss = loss.item()
				acc = model_output["acc"]
				loss_epoch += loss * num_examples
				acc_epoch += acc * num_examples
				num_examples_epoch += num_examples

			if writer:
				writer.add_scalar("test_loss", loss_epoch / num_examples_epoch, global_step)
				writer.add_scalar("test_acc", acc_epoch / num_examples_epoch, global_step)

		return {
			"loss": loss_epoch / num_examples_epoch,
			"acc": acc_epoch / num_examples_epoch,
		}

	num_epochs = 30
	max_gradient_clipping_norm = None  # No gradient clipping.
	#max_gradient_clipping_norm = 1.0

	# Build a model.
	model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # torchvision.models.resnet.ResNet.
	print("Model:", model, sep="\n")
	print(f"#model parameters = {sum(p.numel() for p in model.parameters())}, #trainable model parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}.")

	# Freeze the model parameters.
	for p in model.parameters():
		p.requires_grad = False

	# Replace some layers.
	if False:
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
	else:
		model.fc = torch.nn.Sequential(
			torch.nn.Dropout(0.5),
			torch.nn.Linear(model.fc.in_features, 1024),
			torch.nn.Dropout(0.2),
			torch.nn.Linear(1024, 512),
			torch.nn.Dropout(0.1),
			torch.nn.Linear(512, num_classes),
			#torch.nn.Sigmoid(),
		)
	#print("Model:", model, sep="\n")
	print(f"#model parameters (after model.fc is replaced) = {sum(p.numel() for p in model.parameters())}, #trainable model parameters (after model.fc is replaced) = {sum(p.numel() for p in model.parameters() if p.requires_grad)}.")

	# Initialize the model.
	for param in model.fc.parameters():
		if param.dim() > 1:
			torch.nn.init.xavier_uniform_(param)  # Initialize parameters with Glorot / fan_avg.

	model = model.to(device)

	# Define a criterion.
	criterion = torch.nn.CrossEntropyLoss(reduction="mean")

	# Define an optimizer.
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(params, lr=1e-2)
	# Define a scheduler.
	if False:
		scheduler = None
		is_epoch_based_scheduler = True
	elif False:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
		is_epoch_based_scheduler = True
	else:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader), eta_min=0.0)
		is_epoch_based_scheduler = False

	# Default 'log_dir' is 'runs' - we'll be more specific here.
	#writer = SummaryWriter(log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix="")
	writer = SummaryWriter(log_dir=os.path.join(output_dir_path, "tensorboard_logs"))

	# Train.
	global_step = 0
	for epoch in range(num_epochs):
		start_time = time.time()
		train_results = train_epoch(epoch, global_step, model, criterion, optimizer, scheduler, is_epoch_based_scheduler, train_dataloader, max_gradient_clipping_norm, device, writer)
		global_step = train_results["global_step"]
		val_results = evaluate_model(global_step - 1, model, criterion, val_dataloader, device, writer)
		print(f"Epoch {epoch + 1:02}/{num_epochs:02}: {time.time() - start_time} secs.")
		print(f'\tTrain:      loss = {train_results["loss"]:.3f}, acc = {train_results["acc"]:.3f}.')
		print(f'\tValidation: loss = {val_results["loss"]:.3f}, acc = {val_results["acc"]:.3f}.')

	test_results = test_model(global_step - 1, model, criterion, val_dataloader, device, writer)
	print(f'Test: loss = {test_results["loss"]:.3f}, acc = {test_results["acc"]:.3f}.')

	# Save the trained model.
	model_filepath = os.path.join(output_dir_path, "resnet_dog_cat_fine_tuning.pth")
	save_model(model_filepath, model)

	#--------------------
	# Inference.

	model.eval()

	gts, predictions, images = list(), list(), list()
	with torch.no_grad():
		for batch_inputs, batch_outputs in val_dataloader:
			images.append(batch_inputs.numpy())
			gts.append(batch_outputs.numpy())  # [batch size].
			predictions.append(model(batch_inputs.to(device)).cpu().numpy())  # [batch size, #classes].
	images, gts, predictions = np.vstack(images), np.hstack(gts), np.argmax(np.vstack(predictions), axis=-1)
	assert len(images) == len(gts) == len(predictions)
	num_examples = len(gts)

	results = gts == predictions
	num_correct_examples = results.sum().item()
	acc = results.mean().item()

	print("Prediction: accuracy = {} / {} = {}.".format(num_correct_examples, num_examples, acc))

	for idx, (img, pred, gt) in enumerate(zip(images, predictions, gts)):
		print(f'{pred} (prediction) {"==" if pred == gt else "!="} {gt} (G/T).')

		plt.figure()
		plt.imshow(img.transpose((1, 2, 0)))
		plt.axis("off")
		plt.tight_layout()
		plt.show()

		if idx >= 2:
			break

def main():
	#finetuning_example()
	#convnet_as_fixed_feature_extractor_example()

	resnet_dog_cat_fine_tuning_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
