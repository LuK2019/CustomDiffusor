import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNNValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RNNValueNetwork, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :] 
        x = self.fc(x)
        return x

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.min, self.max = self.calculate_min_max()

    def calculate_min_max(self):
        all_values = [item for traj in self.data for pair in traj[0] for item in pair]
        all_tensor = torch.tensor(all_values, dtype=torch.float32).view(-1, 4)
        min_val = all_tensor.min(dim=0)[0]
        max_val = all_tensor.max(dim=0)[0]
        return min_val, max_val

    def normalize(self, tensor):
        tensor = tensor.view(-1, 4)
        normalized = 2 * (tensor - self.min.view(1, -1)) / (self.max - self.min).view(1, -1) - 1
        return normalized.view(-1)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory, reward = self.data[idx]
        flat_trajectory = torch.tensor([item for pair in trajectory for item in pair], dtype=torch.float32)
        normalized_trajectory = self.normalize(flat_trajectory)
        return normalized_trajectory.view(40, 4), torch.tensor([reward], dtype=torch.float32)



def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def train_value_network(dataset, model, optimizer, criterion, num_epochs=10, batch_size=32):
    model.to(DEVICE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    final_loss = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(dataloader)}')
        final_loss = epoch_loss / len(dataloader)
    
    save_checkpoint(model, optimizer, num_epochs, final_loss, os.path.join('models', 'value_model_final.ckpt'))

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


if __name__ == "__main__":
    dataset_file_path = '/home/bearhaon/MockSimulator/logs/planner_buffer_20231213-055554.pkl' 
    dataset = TrajectoryDataset(load_dataset(dataset_file_path))

    print(torch.cuda.is_available())

    input_dim = 4  
    hidden_dim = 256
    output_dim = 1
    num_layers = 2

    value_model = RNNValueNetwork(input_dim, hidden_dim, output_dim, num_layers)
    value_model.to(DEVICE)  
    optimizer = optim.Adam(value_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Using device: ", DEVICE)
    last_epoch = train_value_network(dataset, value_model, optimizer, criterion, num_epochs=5000, batch_size=32)