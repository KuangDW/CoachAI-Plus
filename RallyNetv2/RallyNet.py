import logging
import os
import fire
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal
import torchsde
import torch.nn.functional as F
from dtaidistance import dtw_ndim
from utilize import *
from tqdm import tqdm, trange
import pickle
import sys
from tqdm import trange
import gc


os.environ['CUDA_LAUNCH_BLOCKING']='1'

SEQ_LEN = 0

with open('Current_dataset/target_players_ids.pkl', 'rb') as f:
    target_players = pickle.load(f)

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class INV(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(INV, self).__init__()
        self.fc = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        shot_types = state[:, :, 12].long()
        player_ids = state[:, :, 17].long()
        shot_embeds = self.shot_embedding(shot_types)
        player_embeds = self.player_embedding(player_ids)
        state = torch.cat((state[:, :, :12], shot_embeds, state[:, :, 13:17], player_embeds), dim=-1)
        return state

    def forward(self, x):
        # x = x.float()
        # raw_out = torch.cat((self.embed_and_transform(x[:, :18]), self.embed_and_transform(x[:, 18:])), -1)
        out = self.fc(x)
        out = self.relu(out)
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)

        return land_logit, shot_logit, move_logit

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, x):
        x = x.float()
        out = self.fc(x)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)
        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]
        return land_logit, shot_logit, move_logit

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, player_ids_len, shot_type_len):
        super(LatentSDE, self).__init__()
        self.action_dim = 5
        self.state_dim = data_size - self.action_dim
        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len
        
        # ================= #
        inverse_model = INV(32, 128, player_ids_len, shot_type_len).cuda(0)
        inverse_model.load_state_dict(torch.load("INV_weight.pth", weights_only=True))
        inverse_model.eval()
        self.inverse_model = inverse_model
        # ================= #

        # ================= #
        self.action_model = MLP(latent_size, 128, player_ids_len, shot_type_len).cuda(0)
        self.ce = nn.CrossEntropyLoss(ignore_index = 0)
        self.mse = nn.MSELoss()        
        # ================= #


        # Encoder.
        self.encoder = Encoder(input_size=self.state_dim, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, self.state_dim)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None


    def SDE_embed_and_transform(self, tensor_data):
        max_length, batch_size, _ = tensor_data.shape
        # Split state and action
        state = tensor_data[:, :, :18]
        action = tensor_data[:, :, 18:]
        state = self.inverse_model.embed_and_transform(state) 
        # Concatenate the state and action back together
        transformed_data = torch.cat((state, action), dim=2)
        return transformed_data

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).
    
    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, total_xs, xs, ts, noise_std, adjoint=False, method="euler", online=False, train_env = None, seeds_env = None):
        global SEQ_LEN
        # data preprocess
        xs = self.SDE_embed_and_transform(xs).float()
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=2/SEQ_LEN, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=2/SEQ_LEN, logqp=True, method=method)

        _xs = self.projector(zs)


        """ MLP training start """
        true_actions = xs[:,:,-self.action_dim:]
        true_land = true_actions[:, :, :2].view(-1, 2)
        true_move = true_actions[:, :, 3:].view(-1, 2)
        true_shot = true_actions[:, :, 2].long().view(-1)

        land, shot, move = self.action_model(zs)

        land_loss = self.mse(land.view(-1, 2), true_land)
        shot_loss = self.ce(shot.view(-1, self.shot_type_len), true_shot)
        move_loss = self.mse(move.view(-1, 2), true_move)
        """ MLP training end """

        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs[:,:,:self.state_dim]).sum(dim=(0, 2)).mean(dim=0)


        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        return log_pxs, logqp0 + logqp_path + land_loss + shot_loss + move_loss

    def predict_actions(self, model, state_sequence):
        actions = []
        type_porb = []
        with torch.no_grad():
            for i in range(len(state_sequence) - 1):
                state1 = state_sequence[i].squeeze()  # 第一个state数组
                state2 = state_sequence[i + 1].squeeze()  # 第二个state数组
                merged_state = torch.cat((state1, state2), dim=0).unsqueeze(0)  # 合并两个state数组
                land_logit, shot_logit, move_logit = model(merged_state)  # 使用模型预测动作
                
                type_porb.append((shot_logit).clone())
                # Predict shot type
                shot_type_probs = F.softmax(shot_logit, dim=-1)
                shot_type_dist = Categorical(shot_type_probs)
                shot_type = shot_type_dist.sample()[0].unsqueeze(0) 

                action = torch.cat((land_logit, shot_type.unsqueeze(0), move_logit),1)
                actions.append(action)
        
        # actions.append(torch.tensor([[[0, 1, 0, 0, 0]]], dtype=torch.float32).cuda(0))
        return torch.cat(actions, dim=0), type_porb

    @torch.no_grad()
    def sample(self, ht, ts, step, player_ids_len, shot_type_len, bm=None):
        global SEQ_LEN
        
        ht = self.SDE_embed_and_transform(ht).float()
        ctx = self.encoder(torch.flip(ht[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[step]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, ts[step:], names={'drift': 'h'}, dt=2/SEQ_LEN, bm=bm)
        
        _xs = self.projector(zs)
        
        action, shot_logit = self.predict_actions(self.inverse_model, _xs) # 所以 forward 時不需要再過一次 embedding 了

        land, shot, move = self.action_model(zs)
        shot_probs = torch.softmax(shot, dim=-1)  # [61, 1, 12]
        shot_sample = torch.multinomial(shot_probs.view(-1, 12), num_samples=1).view(-1, 1, 1)  # [61, 1, 1]
        mlp_action = torch.cat([land, shot_sample.float(), move], dim=-1)  # [61, 1, 5]
        return _xs, action[0,:], shot_logit[0], mlp_action[0,:,:].squeeze(0), shot[0,:,:]

def validate(model, xs, ts, player_ids_len, shot_type_len):
    global SEQ_LEN
    ce_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    model.eval()
    
    total_cross_entropy = 0
    total_ctc_loss = 0
    total_mae_land = 0
    total_mse_land = 0
    total_dtw_land = 0
    total_mae_move = 0
    total_mse_move = 0
    total_dtw_move = 0

    error_list = []


    for rally_index in trange(xs.size(1)):
        rally = xs[:,rally_index,:].unsqueeze(1)

        player_id = rally[0,0,17]
        opponent_id = rally[1,0,17]

        ht = torch.zeros_like(rally[:,:,:18])
        
        initial_state = rally[0, :, :18]
        true_actions = rally[:, :, 18:]

        true_len = find_first_zero_index(rally[:,0,0].tolist())
        if true_len == -1:
            continue

        model_outputs = []
        shot_logits = []
        state = initial_state.clone().squeeze(0)

        step = 0
        done = False
        
        TRANSITION = False
        ROLLOUT = False

        while done == False:
            # print(step)

            ht[step,:,:] = state
            _, inv_action, inv_shot_logit, mlp_action, mlp_shot_logit = model.sample(ht, ts, step, player_ids_len, shot_type_len)
            
            """ inv_output or mlp_output """
            # action = inv_action
            action = mlp_action
            shot_logits.append(mlp_shot_logit) # inv_shot_logit, mlp_shot_logit
            
            model_outputs.append(action)
            step += 1
            
            if TRANSITION == True:
                # Update state for the next step
                if state[17] == player_id:
                    state = transition(action, state, opponent_id)
                else:
                    state = transition(action, state, player_id)
            else:
                state = rally[step,:,:18]
            
            if ROLLOUT == True:
                done = terminate(state)
                if step == SEQ_LEN-1:
                    done = True
            else:
                if step == true_len:
                    done = True

        # print(step, true_len)
        
        model_outputs = torch.stack(model_outputs).unsqueeze(1)
        shot_logits = torch.stack(shot_logits).unsqueeze(1)

        true_landing_pos = true_actions[:, :, :2]
        true_movement_pos = true_actions[:, :, 3:]
        true_shot_type = true_actions[:, :, 2].long().unsqueeze(1)
        pred_landing_pos = model_outputs[:, :, :2]
        pred_movement_pos = model_outputs[:, :, 3:]

        # Calculate errors
        if ROLLOUT != True:
            cross_entropy = ce_fn(shot_logits.view(-1, 12), true_shot_type[:int(shot_logits.size(0))].view(-1)).item()
            mae_landing = F.l1_loss(pred_landing_pos, true_landing_pos[:int(shot_logits.size(0))]).item()
            mse_landing = F.mse_loss(pred_landing_pos, true_landing_pos[:int(shot_logits.size(0))]).item()
            mae_movement = F.l1_loss(pred_movement_pos, true_movement_pos[:int(shot_logits.size(0))]).item()
            mse_movement = F.mse_loss(pred_movement_pos, true_movement_pos[:int(shot_logits.size(0))]).item()
        else:
            cross_entropy = 0
            mae_landing = 0
            mse_landing = 0
            mae_movement = 0
            mse_movement = 0
        
        true_shot_type = true_shot_type.squeeze(1)
        true_shot_type = true_shot_type[true_shot_type != 0]

        # ctc loss
        probs = F.softmax(shot_logits, dim=-1)
        probs = probs.squeeze(1).squeeze(1)
        padding_size = SEQ_LEN - probs.size(0)
        padded_tensor = F.pad(probs, (0, 0, 0, padding_size))
        eps = 1e-8
        padded_tensor[probs.size(0):, 0] = 1 - eps*11
        padded_tensor[probs.size(0):, 1:] = eps
        ctc_loss = F.ctc_loss(torch.log(padded_tensor), true_shot_type, torch.LongTensor([SEQ_LEN]), torch.LongTensor([true_len]), reduction='mean').item()

        # dtw loss
        dist_land = dtw_ndim.distance(pred_landing_pos.squeeze().cpu().numpy(), true_landing_pos.squeeze().cpu().numpy())
        dist_move = dtw_ndim.distance(pred_movement_pos.squeeze().cpu().numpy(), true_movement_pos.squeeze().cpu().numpy())


        # sum up
        total_cross_entropy += cross_entropy
        total_ctc_loss += ctc_loss
        total_mae_land += mae_landing
        total_mse_land += mse_landing
        total_dtw_land += dist_land / true_len
        total_mae_move += mae_movement
        total_mse_move += mse_movement
        total_dtw_move += dist_move / true_len

        error_list.append([cross_entropy, ctc_loss, mae_landing, mse_landing, (dist_land / true_len), mae_movement, mse_movement, (dist_move / true_len)])

    num_rallies = xs.size(1)

    avg_cross_entropy = total_cross_entropy / num_rallies
    avg_ctc_loss = total_ctc_loss / num_rallies
    avg_mae_land = total_mae_land / num_rallies
    avg_mse_land = total_mse_land / num_rallies
    avg_dtw_land = total_dtw_land / num_rallies

    avg_mae_move = total_mae_move / num_rallies
    avg_mse_move = total_mse_move / num_rallies
    avg_dtw_move = total_dtw_move / num_rallies

    with open('error_list.pkl', 'wb') as f:
        pickle.dump(error_list, f)
    
    return avg_cross_entropy, avg_ctc_loss, avg_mae_land, avg_mse_land, avg_dtw_land, avg_mae_move, avg_mse_move, avg_dtw_move

def specify_players(xs, player_name, opponent_name):
    player_id = target_players.index(player_name)
    opponent_id = target_players.index(opponent_name)
    
    player_id_value = xs[:, :, 17]
    mask = (player_id_value == player_id) | (player_id_value == opponent_id)
    filtered_xs = xs[:, mask.any(dim=0), :]
    return filtered_xs


def make_dataset(testing, device, split_ratio=0.8):
    global SEQ_LEN

    with open("Current_dataset/player_train_0.pkl", 'rb') as f:
        player_data = pickle.load(f)
    with open("Current_dataset/opponent_train_0.pkl", 'rb') as f:
        opponent_data = pickle.load(f)

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
    xs = pad_and_convert_to_tensor(merged_data, max_length).cuda(0)
    ts = torch.linspace(0, xs.shape[0], steps=xs.shape[0], device=device)
    
    SEQ_LEN = xs.shape[0]

    min_val = torch.min(ts)
    max_val = torch.max(ts)
    normalized_tensor = (ts - min_val) / (max_val - min_val)
    ts = normalized_tensor * 2

    # Split dataset into training and testing
    total_rallies = xs.shape[1]
    split_index = int(total_rallies * split_ratio)

    if testing == True:
        # Use last 20% for testing
        xs_test = xs[:, split_index:, :]
        # specify_players(xs_test, 'Anders ANTONSEN', 'Kento MOMOTA')
        # xs_test = xs[:, split_index:split_index+100, :]
        return xs_test, ts, len(target_players), len(set(shot_type_set))
    else:
        # Use first 80% for training
        xs_train = xs[:, :split_index, :]
        return xs_train, ts, len(target_players), len(set(shot_type_set))


def main(
        batch_size = 32, 
        latent_size= 32,
        context_size=64,
        hidden_size=64,
        lr_init=1e-4,
        t0=0.,
        t1=2.,
        lr_gamma=0.999,
        num_iters=30000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.1,
        adjoint=False,
        train_dir='./dump/',
        method="euler",
        eval = False, 
        load = False,
        load_step = 0,
        split_ratio = 0.8
):
    create_folder_if_not_exists(train_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if eval == False:
        xs, ts, player_ids_len, shot_type_len = make_dataset(False, device=device, split_ratio=split_ratio)
        
        latent_sde = LatentSDE(
            data_size = 37, # 16 + 8 + 8 + (5)
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
            player_ids_len = player_ids_len,
            shot_type_len = shot_type_len
        ).to(device)

        hyperparameters = {
            "data_size": 37,
            "latent_size": latent_size,
            "context_size": context_size,
            "hidden_size": hidden_size,
            "player_ids_len": player_ids_len,
            "shot_type_len": shot_type_len,
            "ts": ts,
            "xs": xs,
            "target_players": target_players,
        }
        with open('hyperparameters.pkl', 'wb') as f:
            pickle.dump(hyperparameters, f)

        if load == True:
            latent_sde.load_state_dict(torch.load(train_dir+"gen_e_"+str(load_step) +".trc", map_location = device))

        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init) 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        index = 0

        torch.save(latent_sde.state_dict(), train_dir+'gen_e_0.trc')

        for global_step in tqdm(range(1, num_iters + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, xs[:,index:index+batch_size,:], ts, noise_std, adjoint, method, False)

            index += batch_size
            if (index <= int(xs.shape[1]) ) and (index + batch_size >= int(xs.shape[1])):
                shuffled_indices = torch.randperm(xs.size(1))
                xs = xs[:, shuffled_indices, :]
                index = 0

            loss = - log_pxs +  log_ratio * kl_scheduler.val 
            loss.backward()
            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            sys.stdout.flush()
            gc.collect()

            if (global_step % pause_every == 0) or global_step == 1:
                if load == True:
                    torch.save(latent_sde.state_dict(), train_dir+'gen_e_{}.trc'.format(str(load_step + global_step)))
                else:
                    torch.save(latent_sde.state_dict(), train_dir+'gen_e_{}.trc'.format(str(global_step)))
                
                logging.warning(
                    f'global_step: {global_step:06d}, '
                    f'log_pxs: {log_pxs :.4f}, log_ratio: {log_ratio:.4f}, loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}')
    else:
        
        xs, ts, player_ids_len, shot_type_len = make_dataset(True, device=device, split_ratio=split_ratio)
        
        latent_sde = LatentSDE(
            data_size= 37, 
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
            player_ids_len = player_ids_len,
            shot_type_len = shot_type_len
        ).to(device)


        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=t1, size=(1, latent_size,), device=device, levy_area_approximation="space-time")

        latent_sde.load_state_dict(torch.load(train_dir+"gen_e_27000.trc", map_location = device, weights_only=True))

        avg_cross_entropy, avg_ctc_loss, avg_mae_land, avg_mse_land, avg_dtw_land, avg_mae_move, avg_mse_move, avg_dtw_move = validate(latent_sde, xs, ts, player_ids_len, shot_type_len)
        print("Validation results - Cross Entropy:", avg_cross_entropy)
        print("Validation results - CTC Loss:", avg_ctc_loss)
        print("Validation results - Land MAE:", avg_mae_land)
        print("Validation results - Land MSE:", avg_mse_land)
        print("Validation results - Land DTW:", avg_dtw_land)
        print("Validation results - Move MAE:", avg_mae_move)
        print("Validation results - Move MSE:", avg_mse_move)
        print("Validation results - Move DTW:", avg_dtw_move)

if __name__ == "__main__":
    fire.Fire(main)

