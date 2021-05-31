import torch
from torch.optim import Optimizer


class LQA(Optimizer):

	def __init__(self, params, lr_0=1e-5, model=None, loss_fn=None):
		assert lr_0 >= 0
		assert model is not None
		self.model = model
		assert loss_fn is not None
		self.loss_fn = loss_fn
		defaults = dict(lr_0=lr_0)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, loss, input, target, closure=None):
		assert closure is None

		def change(params, lr):
			for p in params:
				p.data += lr * p.grad

		def compute_loss():
			y = self.model(input)
			loss = self.loss_fn(y, target)
			return loss

		for group in self.param_groups:
			params = [ p for p in group['params'] if p.grad is not None ]
			lr_0 = group['lr_0']

			change(params, lr_0)
			pos = compute_loss(params)
			change(params, - 2 * lr_0)
			neg = compute_loss(params)
			a = pos - neg
			b = pos + neg - 2 * loss
			lr = 0.5 * (a / b)
			
			change(lr_0 - lr)