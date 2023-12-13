import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import reset_start_and_target, limits_unnormalizer, load_checkpoint
from train import get_optimizer, get_model, get_noise_scheduler
from value_planner import RNNValueNetwork 


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = "13-12-2023_15-48-17_new_final_step_5000"
VALUE_MODEL_CHECKPOINT = "/home/bearhaon/CustomDiffusor/plan_models/value_model_final.ckpt"  
USE_VALUE_FUNCTION = 1


class SamplingConfig:
  batch_size = 3
  horizon = 40
  state_dim = 2
  action_dim = 2
  learning_rate = 1e-4 # Only relevant to load the optimizer
  eta = 1.0
  num_train_timesteps = 1000
  min = 0
  max = 20

def reshape_and_unpack(x):
    batch_size, num_features, horizon = x.shape
    unpacked_trajectories = []

    for b in range(batch_size):
        trajectory = np.empty((horizon, num_features)) 

        for t in range(horizon):
            action_state = x[b, :, t].cpu().numpy()
            trajectory[t, :] = action_state

        unpacked_trajectories.append(trajectory)

    return unpacked_trajectories


def load_value_model(checkpoint_path):
    value_model = RNNValueNetwork(4, 256, 1, 2)  # Adjust parameters as necessary
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if "model_state_dict" in checkpoint:
        value_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        value_model.load_state_dict(checkpoint)

    value_model.to(DEVICE)
    value_model.eval() 
    return value_model


if __name__ == "__main__":
  config = SamplingConfig()
  shape = (config.batch_size,config.state_dim+config.action_dim, config.horizon)
  scheduler = get_noise_scheduler(config)
  model = get_model("unet1d")
  optimizer = get_optimizer(model, config)
  model, optimizer = load_checkpoint(model, optimizer, "models/"+CHECKPOINT+".ckpt")
  conditions = {
                0: torch.ones((config.batch_size, config.state_dim))*(-0.6),
                -1: torch.ones((config.batch_size, config.state_dim))*0.6, #*torch.tensor([-1, 1])
              }


  if USE_VALUE_FUNCTION:
    value_model = load_value_model(VALUE_MODEL_CHECKPOINT)

  x = torch.randn(shape, device=DEVICE)
  print("Initial noise vector: ", x[0,:,:])

  x = reset_start_and_target(x, conditions, config.action_dim)
  print("Initial noise vector after setting start and target: ", x[0,:,:])

  for i in tqdm.tqdm(scheduler.timesteps):

      timesteps = torch.full((config.batch_size,), i, device=DEVICE, dtype=torch.long)

      with torch.no_grad():
        residual = model(x, timesteps).sample

      obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

      if config.eta > 0:
        noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
        posterior_variance = scheduler._get_variance(i)
        obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * config.eta* noise  # no noise when t == 0

      obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, config.action_dim)
      x = obs_reconstruct_postcond

      if USE_VALUE_FUNCTION and i % 1 == 0:
            trajectories = reshape_and_unpack(x)
            trajectory_tensors = [torch.tensor(traj, dtype=torch.float32, device=DEVICE) for traj in trajectories]

            with torch.no_grad():
                values = torch.stack([value_model(traj_tensor.unsqueeze(0)) for traj_tensor in trajectory_tensors]).squeeze()

            # Select the trajectory with the highest estimated reward
            best_trajectory_index = torch.argmax(values).item()
            x = x[best_trajectory_index].unsqueeze(0).repeat(config.batch_size, 1, 1)

      # convert tensor i to int 
      i = int(i)
      if i < 10 and i%1==0:
        print(f"At step {i}:", x[0,:,:],"\n" , limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max))
        for k in range(1):
          unnormalized_output = limits_unnormalizer(x[k,:,:].cpu(), config.min, config.max)

          # Create a sequence of numbers from 0 to 1, corresponding to each point in the trajectory
          num_points = unnormalized_output.shape[1]
          colors = np.linspace(0, 1, num_points)
          colors[0] = 1
          # plot the output
          plt.ylim(config.min, config.max)
          plt.xlim(config.min, config.max)
          # Use a colormap to map the sequence of numbers to colors (light red to dark red)
          # 'Reds' is a built-in colormap ranging from light to dark red
          plt.scatter(unnormalized_output[2,:], unnormalized_output[3,:], c=colors, cmap='Reds')
          print(i, k)
          # Directory path based on CHECKPOINT
          directory_path = f'guided_samples/{CHECKPOINT}'
          # Check if the directory exists, if not, create it
          if not os.path.exists(directory_path):
            os.makedirs(directory_path)

          file_path = f'{directory_path}/plot_step_{i}_index_{k}.png'
          plt.savefig(file_path)
          plt.close()