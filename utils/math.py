import torch
import math

def normal_entropy(std):
	var = std.pow(2)
	entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
	return entropy.sum(1, keepdim=True)