import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import reset_start_and_target, limits_unnormalizer
from train import get_optimizer, get_model, get_noise_scheduler


# ---------------------------------------------------------- #
# --------------------- CONFIG ----------------------------- #
# ---------------------------------------------------------- #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = "08-12-2023_04-27-51_final_step_50000"

class SamplingConfig:
  batch_size = 32
  horizon = 40
  state_dim = 2
  action_dim = 2
  learning_rate = 1e-4 # Only relevant to load the optimizer
  eta = 1.0
  num_train_timesteps = 1000
  min = 0
  max = 20
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Loaded checkpoint from epoch {} with loss {} at path {}".format(epoch, loss, filepath))
    return model, optimizer

if __name__ == "__main__":
  config = SamplingConfig()
  shape = (config.batch_size,config.state_dim+config.action_dim, config.horizon)
  scheduler = get_noise_scheduler(config)
  model = get_model("unet1d")
  optimizer = get_optimizer(model, config)
  model, optimizer = load_checkpoint(model, optimizer, "models/"+CHECKPOINT+".ckpt")
  conditions = {
                0: torch.ones((config.batch_size, config.state_dim))*(-0.9),
                19: torch.ones((config.batch_size, config.state_dim))*0.9*torch.tensor([-1, 1]),
                -1: torch.ones((config.batch_size, config.state_dim))*0.9*torch.tensor([-1, 1])
              }

  x = torch.randn(shape, device=DEVICE)
  print("Initial noise vector: ", x[0,:,:])

  x = reset_start_and_target(x, conditions, config.action_dim)
  print("Initial noise vector after setting start and target: ", x[0,:,:])

  for i in tqdm.tqdm(scheduler.timesteps):

      timesteps = torch.full((config.batch_size,), i, device=DEVICE, dtype=torch.long)

      with torch.no_grad():
        # print("shape of x and timesteps: ", x.shape, timesteps.shape)
        residual = model(x, timesteps).sample

      obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

      if config.eta > 0:
        noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
        posterior_variance = scheduler._get_variance(i)
        obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * config.eta* noise  # no noise when t == 0

      obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, config.action_dim)
      x = obs_reconstruct_postcond

      if i%500 == 0:
        print(f"At step {i}:", x[0,:,:],"\n" , limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max))
        unnormalized_output = limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max)

        # Create a sequence of numbers from 0 to 1, corresponding to each point in the trajectory
        num_points = unnormalized_output.shape[1]
        colors = np.linspace(0, 1, num_points)\
        # plot the output
        plt.ylim(config.min, config.max)
        plt.xlim(config.min, config.max)
        # Use a colormap to map the sequence of numbers to colors (light red to dark red)
        # 'Reds' is a built-in colormap ranging from light to dark red
        plt.scatter(unnormalized_output[2,:], unnormalized_output[3,:], c=colors, cmap='Reds')
        plt.show()  
