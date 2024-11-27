import torch
import torch.nn as nn

class Encoder(nn.Module):
	"""
	:class param
		input_dim: 输入图片数据维度
		condition_dim: 条件独热编码的维度
		latent_dim: 潜在向量的维度

	:forward param
		x: 输入图片数据
		c: 条件独热编码向量

	:return
		m: 重采样之后的均值
		log: 重采样之后的方差的对数
	"""
	def __init__(self, input_dim, condition_dim, latent_dim):
		super(Encoder, self).__init__()
		self.input_dim = input_dim + condition_dim
		self.enc_mlp = nn.Sequential(
			nn.Linear(self.input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
		)
		self.mean_layer = nn.Linear(128, latent_dim)
		self.log_layer = nn.Linear(128, latent_dim)

	def forward(self, x, c):
		x = torch.cat([x, c], dim=-1) # 拼接图片数据向量和条件对应的独热编码
		z = self.enc_mlp(x)

		# 重参数化
		m = self.mean_layer(z)
		log = self.log_layer(z)

		return m, log

class Decoder(nn.Module):
	"""
	:class param
		latent_dim: 潜在向量的维度
		condition_dim: 条件独热编码的维度
		output_dim: 还原数据的维度

	:forward param
		z: 潜在向量
		c: 条件独热编码向量

	:return
		x_recon: 重建数据
		z: 潜在向量
	"""
	def __init__(self, latent_dim, condition_dim, input_dim):
		super(Decoder, self).__init__()
		self.latent_dim = latent_dim + condition_dim
		self.dec_mlp = nn.Sequential(
			nn.Linear(self.latent_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, input_dim),
			nn.Sigmoid(),
		)

	def forward(self, z, c):
		z = torch.cat([z, c], dim=-1) # 拼接潜在向量和条件对应的独热编码
		x_recon = self.dec_mlp(z)

		return x_recon

class CVAE(nn.Module):
	def __init__(self, input_dim, condition_dim, latent_dim):
		# Encoder & Decoder 初始化
		super(CVAE, self).__init__()
		self.enc = Encoder(input_dim, condition_dim, latent_dim)
		self.dec = Decoder(latent_dim, condition_dim, input_dim)

	def reparameterize(self, mean, log):
		std = torch.exp(0.5 * log)
		eps = torch.randn_like(std)
		return mean + std*eps

	def inference(self, z, c):
		x_recon = self.dec(z, c)

		return x_recon

	def forward(self, x, c):
		m, log = self.enc(x, c)
		z = self.reparameterize(m, log)
		x_recon = self.dec(z, c)

		return x_recon, z, m, log