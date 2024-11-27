import os
import torch
import argparse
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CVAE
from logger import Logger

def main(args):

	# logger setup
	if args.log_dir is not None and not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	logger = Logger(os.path.join(args.dir, "log.txt"))

	# keep the best model parameters according to avg_loss
	tracker = {"epoch" : None, "criterion" : None, "model_params" : None}

	# device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.write(f"we're using :: {device}\n\n")

	# data preparations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x : x.view(-1)) # flatten the 28x28 image to 1D
	])
	mnist = MNIST(args.data_dir, train=True, transform=transform, download=True)
	dataset = DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True)

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
			x, y = x.to(device), y.to(device)
			c = nn.functional.one_hot(y, num_classes=args.num_classes).float() # one-hot encoding

			# update gradients
			optimizer.zero_grad()
			x_recon, _, m, log = model(x, c)
			loss = loss_fn(x_recon, x, m, log)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		# keep the best model parameters according to avg_loss
		avg_loss = epoch_loss / len(dataset)
		# tracker = {"epoch" : None, "criterion" : None, "model_params" : None}
		if tracker["criterion"] is None or avg_loss < tracker["criterion"]:
			tracker["epoch"] = epoch + 1
			tracker["criterion"] = avg_loss
			torch.save(model.state_dict(), args.model_path)
			pass
		logger.write(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}\n")

	# end Training
	logger.write("\n\nTraining completed\n\n")
	logger.write(f"Best Epoch: {tracker['epoch']}, average loss:{tracker['criterion']}\n")
	if tracker["model_params"] is not None:
		logger.write(f"model with the best performance saved to {args.model_path}.")
	# close
	logger.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_classes", type=int, default=10)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--input_size", type=int, default=28*28)
	parser.add_argument("--latent_size", type=int, default=2)
	# parser.add_argument("--print_every", type=int, default=100)
	parser.add_argument("--data_dir", type=str, default="./data")
	parser.add_argument("--log_dir", type=str, default="./log")
	parser.add_argument("--model_path", type=str, default="cvae_model.pth")

	args = parser.parse_args()

	main(args)