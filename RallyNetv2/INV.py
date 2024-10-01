import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import pickle
from utilize import *


os.environ['CUDA_LAUNCH_BLOCKING']='1'

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
state = xs[:,:,:18].cuda()
action = xs[:,:,-5:].cuda()

# 将 state 和 action 数据转换为适合 MLP 输入的形式
sequence_length, batch_size, feature_dim = state.shape
action_dim = action.shape[-1]

# 准备数据集
inputs = []
labels = []

for i in range(sequence_length - 1):
    current_state = state[i].reshape(batch_size, -1)
    next_state = state[i + 1].reshape(batch_size, -1)
    combined_state = torch.cat((current_state, next_state), dim=1)
    inputs.append(combined_state)
    labels.append(action[i])

inputs = torch.stack(inputs).reshape(-1, feature_dim * 2)
labels = torch.stack(labels).reshape(-1, action_dim)


# 過濾掉全零的樣本
non_zero_inputs = (inputs != 0).any(dim=1)
mask = non_zero_inputs
inputs = inputs[mask]
labels = labels[mask]


# 使用 75% 的数据进行训练，25% 的数据进行评估
dataset = TensorDataset(inputs, labels)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)
        return state

    def forward(self, x):
        x = x.float()
        raw_out = torch.cat((self.embed_and_transform(x[:, :18]), self.embed_and_transform(x[:, 18:])), -1)
        out = self.fc(raw_out)
        out = self.relu(out)

        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)

        # 温度缩放应用在shot type预测上
        # shot_logit = shot_logit / 3

        return land_logit, shot_logit, move_logit
    
input_dim = 32
hidden_dim = 128

model = MLP(input_dim, hidden_dim, player_id_len, shot_type_len).cuda()
ce = nn.CrossEntropyLoss(ignore_index = 0)
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:        
        true_landing_pos = labels[:, :2].float()
        true_movement_pos = labels[:, 3:].float()
        true_shot_type = labels[:, 2].long()

        optimizer.zero_grad()
        land_logit, shot_logit, move_logit = model(inputs)
        
        land_loss = mse(land_logit, true_landing_pos)
        shot_loss = ce(shot_logit, true_shot_type)
        move_loss = mse(move_logit, true_movement_pos)

        loss = land_loss + shot_loss + move_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 评估模型
model.eval()
correct_shot = 0
correct_land = 0 
correct_move = 0

from torch.distributions import Categorical, Normal

with torch.no_grad():
    for inputs, labels in test_loader:
        true_landing_pos = labels[:, :2].float()
        true_movement_pos = labels[:, 3:].float()
        true_shot_type = labels[:, 2].long()

        land_logit, shot_logit, move_logit = model(inputs)
        
        # ========================= #
        land_loss = mse(land_logit, true_landing_pos)
        
        shot_loss = ce(shot_logit, true_shot_type)        
        shot_type_probs = F.softmax(shot_logit, dim=-1)
        shot_type_dist = Categorical(shot_type_probs)
        shot_type = shot_type_dist.sample()
        
        move_loss = mse(move_logit, true_movement_pos)
        # ========================= #
        correct_land += land_loss.item()
        correct_shot += shot_loss.item()
        correct_move += move_loss.item()


print('Shot Loss: ', correct_shot/len(test_loader))
print('Land Loss: ', correct_land/len(test_loader))
print('Move Loss: ', correct_move/len(test_loader))

# 假设模型保存在 model.pth 文件中
PATH = "INV_weight.pth"
# 保存模型
torch.save(model.state_dict(), PATH)
