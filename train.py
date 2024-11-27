import os
import torch
import argparse
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import dataloader

from model import CVAE

def main(args):

	# device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"we're using :: {device}")

	# data preparations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x : x.view(-1)) # flatten the 28x28 image to 1D
	])
	mnist = MNIST(args.data_path, train=True, transform=transform, download=True)
	dataset = dataloader.DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True)

	# loss function for cvae
	def loss_fn(x_recon, x, mean, log):
		bce = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
		kl = -0.5 * torch.sum(1 + log - mean.pow(2) - log.exp())
		return (bce + kl) / x.size(0)

	# model setup
	# need to be fixed...
	model = CVAE(input_dim=args.input_size,
				 condition_dim=args.num_classes,
				 latent_dim=args.latent_size).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Training
	model.train()
	for epoch in args.epochs:

		epoch_loss = 0
		for x, y in dataset:
			x.to(device)
			y.to(device)
			c = nn.functional.one_hot(y, num_classes=args.num_classes).float() # one-hot encoding

			# update gradients
			optimizer.zero_grad()
			x_recon, _, m, log = model(x, c)
			loss = loss_fn(x_recon, x, m, log)
			loss.back_ward()
			optimizer.step()

			epoch_loss += loss.item()

		print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(dataset):.6f}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_classes", type=int, default=10)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--input_size", type=int, default=28*28)
	parser.add_argument("--latent_size", type=int, default=2)
	parser.add_argument("--print_every", type=int, default=100)
	parser.add_argument("--data_path", type=str, default="./data")

	args = parser.parse_args()

	main(args)