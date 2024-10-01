import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from utilize import *  # Ensure this module is available and correctly implemented

# Set CUDA for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def make_dataset(device):
    with open("Current_dataset/player_train_0.pkl", 'rb') as f:
        player_data = pickle.load(f)
    with open("Current_dataset/opponent_train_0.pkl", 'rb') as f:
        opponent_data = pickle.load(f)
    with open('Current_dataset/target_players_ids.pkl', 'rb') as f:
        target_players = pickle.load(f)

    merged_data = []
    shot_type_set = []

    for player_rally, opponent_rally in zip(player_data, opponent_data):
        merged_rally = []
        player_len = len(player_rally)
        opponent_len = len(opponent_rally)
        if player_len > 0:
            max_len = max(player_len, opponent_len)
            for i in range(max_len):
                if i < player_len:
                    merged_rally.append(player_rally[i])
                    shot_type_set.append(player_rally[i][12])
                if i < opponent_len:
                    merged_rally.append(opponent_rally[i])
                    shot_type_set.append(opponent_rally[i][12])

            merged_data.append(merged_rally)

    max_length = max(len(rally) for rally in merged_data)
    xs = pad_and_convert_to_tensor(merged_data, max_length).cuda()

    ts = torch.linspace(0, xs.shape[0], steps=xs.shape[0], device=device)
    min_val = torch.min(ts)
    max_val = torch.max(ts)
    normalized_tensor = (ts - min_val) / (max_val - min_val)
    ts = normalized_tensor * 2

    return xs, ts, len(target_players), len(set(shot_type_set))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xs, ts, player_id_len, shot_type_len = make_dataset(device=device)

# Assuming xs has shape [sequence_length, batch_size, feature_dim]
state = xs[:, :, :18].cuda()    # Current state
action = xs[:, :, -5:].cuda()   # Corresponding action

# Prepare dataset: Map from current_state to action
sequence_length, batch_size, feature_dim = state.shape
action_dim = action.shape[-1]

inputs = []
labels = []

for i in range(sequence_length):
    current_state = state[i].reshape(batch_size, -1)      # Shape: [batch_size, 18]
    current_action = action[i].reshape(batch_size, -1)    # Shape: [batch_size, 5]
    inputs.append(current_state)
    labels.append(current_action)

inputs = torch.stack(inputs).reshape(-1, feature_dim)    # Shape: [sequence_length * batch_size, 18]
labels = torch.stack(labels).reshape(-1, action_dim)     # Shape: [sequence_length * batch_size, 5]

# Filter out samples where input is all zeros
non_zero_inputs = (inputs != 0).any(dim=1)
inputs = inputs[non_zero_inputs]
labels = labels[non_zero_inputs]

# Split into training and testing datasets (75% train, 25% test)
dataset = TensorDataset(inputs, labels)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Increased batch size for efficiency

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Embeddings for categorical features within the state
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        # Output layers for different components of the action
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        # Assuming state[:, 12] is shot_type and state[:, 17] is player_id
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)       # Shape: [batch_size, 8]
        player_embeds = self.player_embedding(player_ids)   # Shape: [batch_size, 8]
        
        # Concatenate embeddings with other state features
        # Adjust indices based on your actual state structure
        state_features = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)  # Shape: [batch_size, 32]
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)  # Shape: [batch_size, 32]
        out = self.fc(embedded)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)

        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]

        # Apply temperature scaling to shot type logits if needed
        # shot_logit = shot_logit / 3

        return land_logit, shot_logit, move_logit

# Define model parameters
input_dim = 32  # Updated from 18 to 32 to match embedded state size
hidden_dim = 128

model = MLP(input_dim, hidden_dim, player_id_len, shot_type_len).cuda()

# Define loss functions and optimizer
ce = nn.CrossEntropyLoss(ignore_index=0)  # For categorical shot type
mse = nn.MSELoss()                         # For continuous landing and movement positions
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        # Extract action components from labels
        true_landing_pos = batch_labels[:, :2].float()      # Shape: [batch_size, 2]
        true_shot_type = batch_labels[:, 2].long()          # Shape: [batch_size]
        true_movement_pos = batch_labels[:, 3:].float()    # Shape: [batch_size, 2]

        optimizer.zero_grad()
        land_logit, shot_logit, move_logit = model(batch_inputs)

        # Compute losses
        land_loss = mse(land_logit, true_landing_pos)
        shot_loss = ce(shot_logit, true_shot_type)
        move_loss = mse(move_logit, true_movement_pos)

        loss = land_loss + shot_loss + move_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluating the model
model.eval()
total_shot_loss = 0.0
total_land_loss = 0.0
total_move_loss = 0.0

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        true_landing_pos = batch_labels[:, :2].float()
        true_shot_type = batch_labels[:, 2].long()
        true_movement_pos = batch_labels[:, 3:].float()

        land_logit, shot_logit, move_logit = model(batch_inputs)

        # Compute losses
        land_loss = mse(land_logit, true_landing_pos)
        shot_loss = ce(shot_logit, true_shot_type)
        move_loss = mse(move_logit, true_movement_pos)

        total_land_loss += land_loss.item()
        total_shot_loss += shot_loss.item()
        total_move_loss += move_loss.item()

# Calculate average losses
avg_shot_loss = total_shot_loss / len(test_loader)
avg_land_loss = total_land_loss / len(test_loader)
avg_move_loss = total_move_loss / len(test_loader)

print('Evaluation Results:')
print(f'Shot Loss: {avg_shot_loss:.4f}')
print(f'Land Loss: {avg_land_loss:.4f}')
print(f'Move Loss: {avg_move_loss:.4f}')

# Saving the model
PATH = "BC_weight.pth"
torch.save(model.state_dict(), PATH)
print(f'Model saved to {PATH}')

