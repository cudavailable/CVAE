import os
import torch
from model import CVAE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端

def generate_images(model, device, digit, args):
	model.eval()
	with torch.no_grad():
		condition = torch.nn.functional.one_hot(
			torch.tensor([digit] * args.num_classes), num_classes=10
		).float().to(device)
		z = torch.randn((args.num_classes, 2)).to(device)  # Random latent vectors
		generated_images = model.inference(z, condition)
		generated_images = generated_images.view(-1, 28, 28).cpu().numpy()

		# Plot images
		if args.recon_dir is not None and not os.path.exists(args.recon_dir):
			os.mkdir(args.recon_dir)
		digit_path = os.path.join(args.recon_dir, str(digit))
		if digit_path is not None and not os.path.exists(digit_path):
			os.mkdir(digit_path)

		# plt.figure(figsize=(5, 5))
		plt.figure()
		for i, img in enumerate(generated_images):
			# plt.subplot(1, args.num_classes, i + 1)
			plt.imshow(img, cmap='gray')
			plt.axis('off')
			plt.savefig(os.path.join(digit_path, f'#{i + 1}.png'))
			plt.close()  # 保存后再关闭

		print(f"generated images saved to {args.recon_dir}")
		# plt.show()

def infer(args):
	# device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# model setup
	model = CVAE(input_dim=args.input_size, condition_dim=args.num_classes, latent_dim=args.latent_size).to(device)
	try:
		model.load_state_dict(torch.load(args.model_path))
	except FileNotFoundError:
		print(f"Error: Model file not found at {args.model_path}")
		return
	model.eval()  # 设置模型为评估模式（如不需训练时）
	print(f"Model loaded from {args.model_path}")

	generate_images(model, device, digit=7, args=args)

	pass