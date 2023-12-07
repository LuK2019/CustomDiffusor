from diffusers import DDPMScheduler, UNet1DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from train import get_optimizer, get_model, get_noise_scheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = "06-12-2023_17-25-17"

# ------------ #
#  Parameters  #
# ------------ #

class SamplingConfig:
  batch_size = 32
  horizon = 8
  state_dim = 1
  action_dim = 1
  learning_rate = 1e-4
  eta = 1.0

# ------------ #


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer

def reset_x0(x_in, cond, act_dim):
	for key, val in cond.items():
		x_in[:, key, act_dim:] = val.clone()
	return x_in

if __name__ == "__main__":
  config = SamplingConfig()
  shape = (config.batch_size,config.state_dim+config.action_dim, config.horizon)
  scheduler = get_noise_scheduler()
  model = get_model("unet1d")
  optimizer = get_optimizer(model, config)
  model, optimizer = load_checkpoint(model, optimizer, "models/"+CHECKPOINT+".ckpt")
  conditions = {
                0: torch.zeros((config.batch_size, config.state_dim)),
              }
  # sample random initial noise vector and condition on first state
  x1 = torch.randn(shape, device=DEVICE)
  # x = reset_x0(x1, conditions, config.action_dim)
  x = x1 

  print("Shape of noise vector: ", x.shape)
  print("Initial noise vector: ", x[0,0,:])

  for i in tqdm.tqdm(scheduler.timesteps):

      timesteps = torch.full((config.batch_size,), i, device=DEVICE, dtype=torch.long)

      with torch.no_grad():
        residual = model(x, timesteps).sample #QUESTION: Why permute? Why .sample?

      # 2. use the model prediction to reconstruct an observation (de-noise)
      obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"] #QUESTION: What is happening here? What is the scheduler obejct? What is happening behind the scenes?
      # 3. [optional] add posterior noise to the sample #QUESTION: What is posterior noise? Why would zou add it? Is this correctly implemented?
      if config.eta > 0:
        noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
        posterior_variance = scheduler._get_variance(i) # * noise
        # no noise when t == 0
        # NOTE: original implementation missing sqrt on posterior_variance
        obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * config.eta* noise  # MJ had as log var, exponentiated

      # 4. apply conditions to the trajectory
      # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, config.action_dim)
      obs_reconstruct_postcond = obs_reconstruct
      x = obs_reconstruct_postcond
      if i%10 == 0:
        print(f"At step {i}:", x[0,0,:])

  