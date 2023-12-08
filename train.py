from tqdm.auto import tqdm
import torch
from diffusers import UNet1DModel
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from utils import reset_start_and_target, limits_normalizer
import pickle
# ------------ #
#  Parameters  #
# ------------ #

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRED_NOISE = False
PATH_TO_PICKLE_FILE = '/Users/luke/Projects/Ocean/MockSimulator/logs/concatenated_buffer_20231202-115430.pkl'

print("Using device: ", DEVICE)

SEED = 0

class TrainingConfig:
    num_epochs = 200 # int(2e6)
    batch_size = 32 # Must be multiple of 8
    learning_rate =1e-5 # 2e-4
    lr_warmup_steps = 1000
    num_train_timesteps = 1000
    horizon = 24 # Must be multiple of 8
    action_dim = 2
    state_dim = 2
# ------------ #

class MockDataset(Dataset):
    def __init__(self, num_samples, sequence_length, num_features):
        self.data = torch.arange(start=0, end=sequence_length, step=1).repeat(num_samples, num_features, 1).float()
        shape = self.data.shape
        self.data = limits_normalizer(self.data)
        assert self.data.shape == shape, f"Normalization changed the shape of the data from {shape} to {self.data.shape}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class SimulatorDataset(Dataset):
    def __init__(self, path_to_pickl_file):
        # load the np.array from the pickle file
        with open(path_to_pickl_file, 'rb') as f:
            self.data = pickle.load(f)
        assert self.data.shape[1] == 4, f"Data must have 4 channels, but has {self.data.shape[1]}"
        print("Loaded data with shape: ", self.data.shape)
        print("Example: First 5 steps of the first trajectory data[0, :, :5]: ", self.data[0, :, :5])
        if self.data.shape[0] % 8 != 0:
            print("Warning: The number of trajectories is not a multiple of 8. The last few trajectories will be ignored.")
            self.data = self.data[:-(self.data.shape[0]%8), :, :]
            print("Number of trajectories after removing the last few: ", self.data.shape[0])
        if self.data.shape[2] % 8 != 0:
            print("Warning: The number of timesteps is not a multiple of 8. The last few timesteps will be ignored.")
            self.data = self.data[:, :, :-(self.data.shape[2]%8)]
            print("Number of timesteps after removing the last few: ", self.data.shape[2])
        # Convert the data to float double
        self.data = self.data.astype('float32')
        # Normalize the data
        shape = self.data.shape
        self.data = limits_normalizer(self.data)
        assert self.data.shape == shape, f"Normalization changed the shape of the data from {shape} to {self.data.shape}"
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.relu(self.batchnorm(self.conv2(x)))
        return x

class DebugNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DebugNet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Decoder
        self.dec1 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

        # Pooling and Upsample
        self.pool = nn.MaxPool1d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        assert x.shape[1] == 2
        assert x.shape[2] == 8

        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        x = self.enc4(x)

        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec1(x)
        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec3(x)
        x = self.final_conv(x)
        return x


def get_model(type="unet1d"):
    '''
        Initializes and sets up the U-Net model.
    '''
    if type == "unet1d":
        model = UNet1DModel(
            sample_size = 24, #TODO: Still not sure how to update this and when.
            sample_rate = None,
            in_channels= 4, # Update depending on the number of channels in the data (i.e., dim states + dim actions)
            out_channels= 4, # Update depending on the number of channels in the data (i.e., dim states + dim actions)
            extra_in_channels= 0,
            time_embedding_type = "positional",
            flip_sin_to_cos = True,
            use_timestep_embedding = True,
            freq_shift = 0.0,
            down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
            mid_block_type = "MidResTemporalBlock1D",
            out_block_type = ("OutConv1DBlock"),
            block_out_channels = (32, 64, 128, 256),
            act_fn= 'mish',
            norm_num_groups= 8,
            layers_per_block= 1,
            downsample_each_block = False,
        ).to(DEVICE)
    elif type == "debug":
        model = DebugNet(in_channels=2, out_channels=2)
    else:
        raise ValueError("Model type not supported")
    return model

def get_noise_scheduler(config):
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, prediction_type="sample")
    return noise_scheduler

def get_optimizer(model, config):
    '''
        Initializes and sets up the optimizer.
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return optimizer

def get_lr_scheduler(optimizer, train_dataloader):
    '''
        Initializes and sets up the learning rate scheduler.
    '''

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    return lr_scheduler

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

def train_loop(config, model, noise_scheduler, optimizer, train_dataset, lr_scheduler, conditions):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    global_step = 0

    model = model.to(DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    
    print(20*"-")
    print("Starting training loop")
    print("Total number of epochs: ", config.num_epochs)
    print("Number of training steps per epoch: ", len(train_dataset)/config.batch_size)
    print(20*"-")

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, clean_trajectories in enumerate(train_dataloader):
            noise = torch.randn(clean_trajectories.shape, device=clean_trajectories.device)
            batch_size = clean_trajectories.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=clean_trajectories.device, dtype=torch.int64)
            noisy_trajectories = noise_scheduler.add_noise(clean_trajectories, noise, timesteps)
            # Print a clean and noisy trajectory
            # if step%500 == 0:
            #     print("Timestep: ", timesteps[0])
            #     print("Clean trajectory", clean_trajectories[0])
            #     print("Noisy trajectory", noisy_trajectories[0])

            clean_trajectories = clean_trajectories.to(DEVICE)
            noisy_trajectories = noisy_trajectories.to(DEVICE)
            timesteps = timesteps.to(DEVICE)
            noise = noise.to(DEVICE)

            pred = model(noisy_trajectories, timesteps, return_dict=False)[0]


            if global_step%10 == 0:
                # get the index of the smallest timestep
                idx = torch.argmin(timesteps)
                idx = idx.to(DEVICE)
                print("Timestep: ", timesteps[idx])
                print("Noisy trajectory", noisy_trajectories[idx])
                print("Prediction", pred[idx])
                print("Learning rate", lr_scheduler.get_last_lr()[0])
            
            if PRED_NOISE:
                loss = F.mse_loss(pred, noise)
            else:
                pred = reset_start_and_target(pred, conditions, config.action_dim)
                # print("Shape of pred after resetting start and target: ", pred.shape)
                loss = F.mse_loss(pred, clean_trajectories)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

            if global_step % 10000 ==0:
                    path = f"models/{dt_string}_step_{global_step}.ckpt"
                    save_checkpoint(model, optimizer, epoch, loss, path)

    now = datetime.now()
    path = f"models/{dt_string}_final_step_{global_step}.ckpt"
    save_checkpoint(model, optimizer, epoch, loss, path)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    config = TrainingConfig()
    model = get_model('unet1d')
    # train_dataloader = MockDataset(num_samples=1600, sequence_length=config.horizon, num_features=config.state_dim+config.action_dim)
    train_dataloader = SimulatorDataset(PATH_TO_PICKLE_FILE)
    noise_scheduler = get_noise_scheduler(config)
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_lr_scheduler(optimizer, train_dataloader)
    conditions = { #TODO: Depending on how we sample, this might be required during traiing as well (as per MJ implementation)
                # 0: torch.zeros((config.batch_size, config.state_dim)),
                #-1: torch.ones((config.batch_size, config.state_dim))
              }
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, conditions)
