#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import functools,time
import torch, torchvision

def box_iou_test():
	box_mode = "xyxy"
	#box_mode = "xywh"

	if box_mode == "xyxy":  # (x1, y1, x2, y2).
		boxes1 = torch.tensor([
			[0.25, 0.3, 0.55, 0.5], [0.25, 0.3, 0.55, 0.5], [0.25, 0.3, 0.55, 0.5],
			[0.2, 0.45, 0.6, 0.75], [0.2, 0.45, 0.6, 0.75], [0.2, 0.45, 0.6, 0.75]
		])
		boxes2 = torch.tensor([
			[0.35, 0.3, 0.65, 0.5], [0.4, 0.3, 0.7, 0.5], [0.45, 0.3, 0.75, 0.5],
			[0.2, 0.35, 0.6, 0.65], [0.2, 0.3, 0.6, 0.6], [0.2, 0.25, 0.6, 0.55]
		])

		assert boxes1.shape[0] == boxes1.shape[0], f'{boxes1.shape[0]} != {boxes1.shape[0]}.'
		box_mask = torch.ones([boxes1.shape[0], 1], dtype=torch.bool)
		box_mask[0, 0] = False
	elif box_mode == "xywh":  # (cx, cy, w, h).
		boxes1 = torch.tensor([
			[0.4, 0.4, 0.3, 0.2], [0.4, 0.4, 0.3, 0.2], [0.4, 0.4, 0.3, 0.2],
			[0.4, 0.6, 0.4, 0.3], [0.4, 0.6, 0.4, 0.3], [0.4, 0.6, 0.4, 0.3]
		])
		boxes2 = torch.tensor([
			[0.5, 0.4, 0.3, 0.2], [0.55, 0.4, 0.3, 0.2], [0.6, 0.4, 0.3, 0.2],
			[0.4, 0.5, 0.4, 0.3], [0.4, 0.45, 0.4, 0.3], [0.4, 0.4, 0.4, 0.3]
		])
		# (cx, cy, w, h) -> (x1, y1, x2, y2).
		boxes1 = torch.stack([boxes1[...,0] - boxes1[...,2] / 2, boxes1[...,1] - boxes1[...,3] / 2, boxes1[...,0] + boxes1[...,2] / 2, boxes1[...,1] + boxes1[...,3] / 2], axis=-1)  # (x1, y1, x2, y2).
		boxes2 = torch.stack([boxes2[...,0] - boxes2[...,2] / 2, boxes2[...,1] - boxes2[...,3] / 2, boxes2[...,0] + boxes2[...,2] / 2, boxes2[...,1] + boxes2[...,3] / 2], axis=-1)  # (x1, y1, x2, y2).

		assert boxes1.shape[0] == boxes1.shape[0], f'{boxes1.shape[0]} != {boxes1.shape[0]}.'
		box_mask = torch.ones([boxes1.shape[0], 1], dtype=torch.bool)
		box_mask[0, 0] = False
	else:
		raise ValueError(f"Invalid box mode: {box_mode}.")
	boxes1 = boxes1 * box_mask
	boxes2 = boxes2 * box_mask

	#-----
	# (x1, y1, x2, y2). 0 <= x1 < x2 and 0 <= y1 < y2.
	# (N, 4) & (M, 4) -> (N, M).
	#box_iou_functor = torchvision.ops.box_iou
	#box_iou_functor = torchvision.ops.generalized_box_iou
	#box_iou_functor = functools.partial(torchvision.ops.distance_box_iou, eps=1e-07)
	box_iou_functor = functools.partial(torchvision.ops.complete_box_iou, eps=1e-07)

	print("Computing box IoU scores...")
	start_time = time.time()
	box_iou_scores = [box_iou_functor(bbox1.unsqueeze(dim=0), bbox2.unsqueeze(dim=0)) for bbox1, bbox2 in zip(boxes1, boxes2)]  # Slower.
	#box_iou_scores = [box_iou_functor(bbox1.unsqueeze(dim=0), bbox2.unsqueeze(dim=0)).item() for bbox1, bbox2 in zip(boxes1, boxes2)]  # Slower.
	print(f"Box IoU scores computed: {time.time() - start_time} sec.")
	print(f"{box_iou_scores=}.")

	print("Computing box IoU scores (all pairs)...")
	start_time = time.time()
	box_iou_scores = box_iou_functor(boxes1, boxes2)
	#box_iou_scores = box_iou_functor(boxes1.contiguous(), boxes2.contiguous())
	#box_iou_scores = box_iou_functor(boxes1.view(-1, 4), boxes2.view(-1, 4))
	print(f"Box IoU scores (all pairs) computed: {time.time() - start_time} sec.")
	print(f"{box_iou_scores=}.")
	print(f"{box_iou_scores.nanmean().item()=}.")
	print(f"{torch.diag(box_iou_scores)=}.")
	print(f"{torch.diag(box_iou_scores).nanmean().item()=}.")

def box_iou_loss_test():
	box_mode = "xyxy"
	#box_mode = "xywh"

	if box_mode == "xyxy":  # (x1, y1, x2, y2).
		boxes1 = torch.tensor([
			[0.25, 0.3, 0.55, 0.5], [0.25, 0.3, 0.55, 0.5], [0.25, 0.3, 0.55, 0.5],
			[0.2, 0.45, 0.6, 0.75], [0.2, 0.45, 0.6, 0.75], [0.2, 0.45, 0.6, 0.75]
		])
		boxes2 = torch.tensor([
			[0.35, 0.3, 0.65, 0.5], [0.4, 0.3, 0.7, 0.5], [0.45, 0.3, 0.75, 0.5],
			[0.2, 0.35, 0.6, 0.65], [0.2, 0.3, 0.6, 0.6], [0.2, 0.25, 0.6, 0.55]
		])

		assert boxes1.shape[0] == boxes1.shape[0], f'{boxes1.shape[0]} != {boxes1.shape[0]}.'
		box_mask = torch.ones([boxes1.shape[0], 1], dtype=torch.bool)
		box_mask[0, 0] = False
	elif box_mode == "xywh":  # (cx, cy, w, h).
		boxes1 = torch.tensor([
			[0.4, 0.4, 0.3, 0.2], [0.4, 0.4, 0.3, 0.2], [0.4, 0.4, 0.3, 0.2],
			[0.4, 0.6, 0.4, 0.3], [0.4, 0.6, 0.4, 0.3], [0.4, 0.6, 0.4, 0.3]
		])  # (cx, cy, w, h).
		boxes2 = torch.tensor([
			[0.5, 0.4, 0.3, 0.2], [0.55, 0.4, 0.3, 0.2], [0.6, 0.4, 0.3, 0.2],
			[0.4, 0.5, 0.4, 0.3], [0.4, 0.45, 0.4, 0.3], [0.4, 0.4, 0.4, 0.3]
		])  # (cx, cy, w, h).
		# (cx, cy, w, h) -> (x1, y1, x2, y2).
		boxes1 = torch.stack([boxes1[...,0] - boxes1[...,2] / 2, boxes1[...,1] - boxes1[...,3] / 2, boxes1[...,0] + boxes1[...,2] / 2, boxes1[...,1] + boxes1[...,3] / 2], axis=-1)
		boxes2 = torch.stack([boxes2[...,0] - boxes2[...,2] / 2, boxes2[...,1] - boxes2[...,3] / 2, boxes2[...,0] + boxes2[...,2] / 2, boxes2[...,1] + boxes2[...,3] / 2], axis=-1)

		assert boxes1.shape[0] == boxes1.shape[0], f'{boxes1.shape[0]} != {boxes1.shape[0]}.'
		box_mask = torch.ones([boxes1.shape[0], 1], dtype=torch.bool)
		box_mask[0, 0] = False
	else:
		raise ValueError(f"Invalid box mode: {box_mode}.")
	boxes1 = boxes1 * box_mask
	boxes2 = boxes2 * box_mask

	#-----
	# (x1, y1, x2, y2). 0 <= x1 < x2 and 0 <= y1 < y2.
	# (N, 4) & (N, 4) -> (N,).
	#box_iou_loss_functor = functools.partial(torchvision.ops.generalized_box_iou_loss, reduction="none", eps=1e-07)
	#box_iou_loss_functor = functools.partial(torchvision.ops.distance_box_iou_loss, reduction="none", eps=1e-07)
	box_iou_loss_functor = functools.partial(torchvision.ops.complete_box_iou_loss, reduction="none", eps=1e-07)

	print("Computing box IoU losses...")
	start_time = time.time()
	box_iou_losses = box_iou_loss_functor(boxes1, boxes2)
	#box_iou_losses = box_iou_loss_functor(boxes1.contiguous(), boxes2.contiguous())
	#box_iou_losses = box_iou_loss_functor(boxes1.view(-1, 4), boxes2.view(-1, 4))
	print(f"Box IoU losses computed: {time.time() - start_time} sec.")
	print(f"{box_iou_losses=}.")
	print(f"{box_iou_losses.nanmean().item()=}.")

def main():
	box_iou_test()
	box_iou_loss_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
