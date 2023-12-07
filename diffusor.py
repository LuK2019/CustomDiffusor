import copy
import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet1DModel

import torch
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EMA:
    '''
        Empirical Moving Average
    '''
    def __init__(self, beta):
        self.beta = beta
        self.avg = None

    def update(self, model):
        if self.avg is None:
            self.avg = copy.deepcopy(model.state_dict())
        else:
            for (k, avg_param), param in zip(self.avg.items(), model.parameters()):
                if param.requires_grad:
                    avg_param.mul_(self.beta).add_(param.data, alpha=1 - self.beta)

class Trainer:
    '''
        Trainer class for the diffusion model.
    '''
    def __init__(self, diffusion_model, dataset, optimizer, device=DEVICE, **kwargs):
        self.model = diffusion_model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = device
        self.kwargs = kwargs

        # Additional attributes based on provided kwargs
        self.ema_decay = kwargs.get('ema_decay', 0.995)
        self.batch_size = kwargs.get('train_batch_size', 32)
        self.gradient_accumulate_every = kwargs.get('gradient_accumulate_every', 2)
        self.log_freq = kwargs.get('log_freq', 100)
        self.save_freq = kwargs.get('save_freq', 1000)
        self.results_folder = kwargs.get('results_folder', './results')

        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0
        )

        # Initialize EMA
        self.ema = EMA(self.ema_decay)

        # Initialize step
        self.step = 0

    def setup_model(self):
        '''
            Initializes and sets up the U-Net model.
        '''
        self.model = UNet1DModel(
            sample_size=None,
            sample_rate=None, #WHAT DOES THAT MEAN?
            in_channels=2, #Adjust to training data (14 for hopper data)
            out_channels=2,
            extra_in_channels=0,    
            time_embedding_type='fourier',
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            freq_shift=0.0,
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            up_block_types=('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1DNoSkip', 'UpBlock1DNoSkip'),
            mid_block_type='UNetMidBlock1D',
            out_block_type=None,
            block_out_channels=(32, 64, 128, 256), # "block_out_channels": [32, 64, 128, 256],
            act_fn=None,
            norm_num_groups=8,
            layers_per_block=1,
            downsample_each_block=False
        ).to(self.device)

    def train_step(self, batch, timestep):
        '''
            Performs a single training step.
        '''
        self.model.train()
        batch = batch.to(self.device)
        print(f"Batch has shape {batch.shape}")
        loss = self.model(batch, timestep=timestep).mean()  # You may need to adjust this based on your model's output
        loss.backward()
        return loss.item()

    def train(self, n_train_steps):
        '''
            Executes the training loop.
        '''
        for step in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = next(iter(self.dataloader))
                loss = self.train_step(batch, step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if step % self.log_freq == 0:
                print(f"Step: {step}, Loss: {loss}")

            if step % self.save_freq == 0:
                self.save_model(step)

            self.step += 1

    def save_model(self, step):
        '''
            Saves the model state.
        '''
        save_path = os.path.join(self.results_folder, f'model_state_{step}.pth')
        torch.save(self.model.state_dict(), save_path)

    # Additional methods for loading model, evaluating, etc., can be added as needed



class MockDataset(Dataset):
    def __init__(self, num_samples=100, sequence_length=50, num_features=2):
        self.data = torch.randn(num_samples, sequence_length, num_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # Instantiate the mock dataset
    mock_dataset = MockDataset()


    # Define the model, optimizer, and other configurations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet1DModel(...) # Initialize the model with appropriate parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize the Trainer
    trainer = Trainer(
        diffusion_model=model,
        dataset=mock_dataset,
        optimizer=optimizer,
        device=device,
        ema_decay=0.995,
        train_batch_size=32,
        gradient_accumulate_every=2,
        log_freq=10,
        save_freq=50,
        results_folder='./results'
    )

    # Set up the model
    trainer.setup_model()

    # Start the training loop
    trainer.train(n_train_steps=100)
