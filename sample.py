from diffusers import DDPMScheduler, UNet1DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from utils import reset_start_and_target, limits_unnormalizer
from train import get_optimizer, get_model, get_noise_scheduler
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = "07-12-2023_16-04-59_final_step_600"

# ------------ #
#  Parameters  #
# ------------ #

class SamplingConfig:
  batch_size = 32
  horizon = 24
  state_dim = 2
  action_dim = 2
  learning_rate = 1e-4 # Only relevant to load the optimizer
  eta = 1.0
  num_train_timesteps = 1000
  min = 0
  max = 20

# ------------ #


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
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
                -1: torch.ones((config.batch_size, config.state_dim))*0.9
              }
  # sample random initial noise vector and condition on first state
  x = torch.randn(shape, device=DEVICE)
  print("Initial noise vector: ", x[0,:,:])

  x = reset_start_and_target(x, conditions, config.action_dim)
  print("Initial noise vector after setting start and target: ", x[0,:,:])

  for i in tqdm.tqdm(scheduler.timesteps):

      timesteps = torch.full((config.batch_size,), i, device=DEVICE, dtype=torch.long)

      with torch.no_grad():
        print("shape of x and timesteps: ", x.shape, timesteps.shape)
        residual = model(x, timesteps).sample

      obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

      if config.eta > 0:
        noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
        posterior_variance = scheduler._get_variance(i)
        # no noise when t == 0
        obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * config.eta* noise  #\\

      obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, config.action_dim)
      x = obs_reconstruct_postcond
      if i%250 == 0 or i == 1:
        print(f"At step {i}:", x[0,:,:],"\n" , limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max))
        unnormalized_output = limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max)
        # plot the output
        # fix the x and y axis to be in the range [min, max]
        plt.ylim(config.min, config.max)
        plt.xlim(0, config.horizon)
        # plot the states as as scatter plot
        plt.scatter(unnormalized_output[2,:],unnormalized_output[3,:])
        if DEVICE == 'cuda':
          plt.savefig(f"CHECKPOINT_sample_step{i}_.png")
        else:
          plt.show()

  
