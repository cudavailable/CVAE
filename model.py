import torch
import torch.nn as nn

class Encoder(nn.Module):
	"""
	:param
		input_dim: 输入图片数据维度
		condition_dim: 条件独热编码的维度
		latent_dim: 潜在向量的维度

	:return
		m: 重采样之后的均值
		log: 重采样之后的方差的对数
	"""
	def __int__(self, input_dim, condition_dim, latent_dim):
		super.__init__(Encoder, self)
		self.input_dim = input_dim + condition_dim
		self.enc = nn.Sequential(
			nn.Linear(self.input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
		)
		self.mean_layer = nn.Linear(128, latent_dim)
		self.log_layer = nn.Linear(128, latent_dim)

	def forward(self, x, c):
		x = torch.concatenate([x, c], dim=-1) # 拼接图片数据向量和条件对应的独热编码
		z = self.enc(x)

		# 重参数化
		m = self.mean_layer(z)
		log = self.log_layer(z)

		return m, log

class Decoder(nn.Module):
	"""
	:param
		latent_dim: 潜在向量的维度
		condition_dim: 条件独热编码的维度
		output_dim: 还原数据的维度

	:return
		x_recon: 重建数据
		z: 潜在向量
	"""
	def __init__(self, latent_dim, condition_dim, output_dim):
		super.__init__(Decoder, self)
		self.latent_dim = latent_dim + condition_dim
		self.dec = nn.Sequential(
			nn.Linear(self.latent_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, output_dim),
			nn.Sigmoid(),
		)

	def forward(self, z, c):
		z = torch.concatenate([z, c], dim=-1) # 拼接潜在向量和条件对应的独热编码
		x_recon = self.dec(z)

		return x_recon, z
