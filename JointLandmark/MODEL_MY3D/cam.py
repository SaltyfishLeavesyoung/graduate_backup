# https://github.com/ahtwq/grad-cam/blob/master/cam.py
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


class FeatureExtractor():
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []
		"""for name, module in self.model._modules['module'].named_children():
			print(name)
			print(len(x))
			if name == 'fc':
				x = x.view(x.size(0), -1)
			x = module(x)

			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
		return outputs, x"""
		x_final, out = self.model(x)
		out = out.view(out.size(0), -1)
		self.gradients = self.model.module.gradients
		outputs += [out]
		return x_final, out


class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		return target_activations, output


def preprocess_image(img):
	test_transform =  transforms.Compose([
					transforms.Resize((896, 896)),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					])

	preprocessed_img = test_transform(img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad=True)
	return input


def show_cam_on_image(img, mask, ID, index):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	#cam = 0.5*heatmap + 0.5 *np.float32(img) / 255
	cam = heatmap
	cam = cam / np.max(cam)
	cv2.imwrite("./res2/cam_4/cam_{}_{}.png".format(ID, index), np.uint8(255 * cam))


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index=None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		"""if index == None:
			index = np.argmax(output.cpu().data.numpy())"""

		#print(output.size())

		"""one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot).reshape(output.shape), requires_grad=True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)"""

		"""one_hot = output.cpu().data.numpy()
		one_hot = Variable(torch.from_numpy(one_hot).reshape(output.shape), requires_grad=True)
		self.model.zero_grad()
		one_hot.backward(torch.ones_like(one_hot), retain_graph=True)"""

		grads = []

		for idx in range(3):
			one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
			one_hot[0][idx] = 1
			one_hot = Variable(torch.from_numpy(one_hot).reshape(output.shape), requires_grad=True)
			if self.cuda:
				one_hot = torch.sum(one_hot.cuda() * output)
			else:
				one_hot = torch.sum(one_hot * output)
			self.model.zero_grad()
			one_hot.backward(retain_graph=True)
	
			grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
			wei = np.mean(grads_val, axis = 2)[0, :]
			grads.append(wei)


		# print(grads_val.shape) (1, 6, 8, 256, 256)

		# print(features.shape) torch.Size([1, 6, 8, 256, 256])

		target = features
		target = np.sum(target.cpu().data.numpy()[0, :], axis = 1)

		# weights = np.mean(grads_val, axis = 1)[0, :]
		weights = np.mean(np.array(grads), axis = 0)

		# weights = np.mean(weights, axis = 0)

		cam = np.zeros(target.shape[1:], dtype=np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]
		# cam = weights

		cam = np.maximum(cam, 0)
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		# print(cam.shape)
		# cam = cv2.resize(cam, (8, 256, 256))

		return cam
