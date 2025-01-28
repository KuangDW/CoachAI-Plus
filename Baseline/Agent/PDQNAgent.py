import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


from Agent.BaseAgent import BaseAgent
from Net.Net import SimpleDQN, SimpleParamNet
from ReplayBuffer.ReplayBuffer import ReplayMemory
from utils import hard_update_target_network, soft_update_target_network, get_prob_from_q


class PDQNAgent(BaseAgent):

    def __init__(self, base_param, hyper_param):
        super().__init__(base_param)
        ### parameter copying
        self.num_episodes = hyper_param.num_episodes
        # self.epsilon_decay = hyper_param.epsilon_decay
        self.epsilon_final = hyper_param.epsilon_final
        self.replay_memory_size = hyper_param.replay_memory_size
        self.batch_size = hyper_param.batch_size
        self.gamma = hyper_param.gamma
        self.lr_q = hyper_param.lr_q
        self.lr_param = hyper_param.lr_param
        self.critic_hidden_layers = hyper_param.critic_hidden_layers
        self.actor_hidden_layers = hyper_param.actor_hidden_layers
        self.warmup_steps = hyper_param.warmup_steps
        self.update_behavior_freq = hyper_param.update_behavior_freq
        self.update_target_freq = hyper_param.update_target_freq
        self.embed_dim = hyper_param.embed_dim
        self.tau_param = hyper_param.tau_param
        self.state_dim = hyper_param.state_dim
        self.action_dim = hyper_param.action_dim
        self.action_param_dim = hyper_param.action_param_dim
        self.sigma = hyper_param.sigma
        self.max_grad_norm = hyper_param.max_grad_norm

        ### model construction
        self.behavior_q_net = SimpleDQN(
            self.state_dim, self.action_param_dim, self.action_dim, self.critic_hidden_layers
        )
        self.target_q_net = SimpleDQN(
            self.state_dim, self.action_param_dim, self.action_dim, self.critic_hidden_layers
        )
        self.behavior_param_net = SimpleParamNet(
            self.state_dim, self.action_param_dim, self.actor_hidden_layers
        )
        self.target_param_net = SimpleParamNet(
            self.state_dim, self.action_param_dim, self.actor_hidden_layers
        )
        self.behavior_q_net = self.behavior_q_net.to(self.device)
        self.target_q_net = self.target_q_net.to(self.device)
        self.behavior_param_net = self.behavior_param_net.to(self.device)
        self.target_param_net = self.target_param_net.to(self.device)

        ### copying parameters
        hard_update_target_network(self.behavior_q_net, self.target_q_net)
        hard_update_target_network(self.behavior_param_net, self.target_param_net)

        ### replay buffer
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        ### initialize epsilon value
        self.epsilon = 1.0
        self.q_optimizer = torch.optim.Adam(self.behavior_q_net.parameters(), lr=self.lr_q, eps=1.5e-4)
        self.param_optimizer = torch.optim.Adam(self.behavior_param_net.parameters(), lr=self.lr_param, eps=1.5e-4)
        
        self.total_timestamp = 0
        self.total_rally = 0

    def act(self, state, launch, eval=False):
        """
        Determines the action to take in the given state.

        :param state:
        :return:
        """
        with torch.no_grad():
            
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            action_param = self.behavior_param_net(state_tensor)

            if launch == True:
                mask = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
            else:
                mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)

            if not eval and random.random() < self.epsilon:
                ### exploration
                #### discrete action
                if launch:
                    action_type = random.randint(0, 1)
                else:
                    action_type = random.randint(2, 9)

                #### continuous action -1 ~ 1
                action_param = action_param.uniform_(-1, 1)
                q_values = self.behavior_q_net(state_tensor, action_param).squeeze(0)
                action_prob = get_prob_from_q(q_values, mask)
                
            else:
                ### greedy
                q_values = self.behavior_q_net(state_tensor, action_param).squeeze(0)
                action_prob = get_prob_from_q(q_values, mask)
                action_type = torch.argmax(action_prob).cpu().item()
            
        action_prob = action_prob.tolist()
        action_param = action_param[0].cpu().tolist()
        land, move = tuple(action_param[:2]), tuple(action_param[2:])
        
        # decode the action from [-1, 1] to correct range
        land, move = self.decode_positions(land, move)

        # action_type is 1 ~ 10
        return action_type+1, land, move, action_prob

    def train(self):

        for rally_idx in tqdm(range(self.num_episodes)):
            
            rally_reward = 0

            state, info, done, launch = self.train_env.reset()

            while not done:

                # reorganize a state with launch and get list format
                new_state = self._encode_state(state, launch)
                action_type, land, move, action_prob = self.act(new_state, launch, eval=False)

                action = (action_type, state[-1], land, move, action_prob)

                if len(info['state'][1:]) >= 60: # Too long
                    break

                next_state, reward, info, done, next_launch = self.train_env.step(action, launch)
                
                # normalize the positions
                _, _, _, norm_land, norm_move = self.encode_positions((0,0), (0,0), (0,0), land, move)

                
                if done:
                    next_state = state

                # reorganize a state with launch and get list format
                new_next_state = self._encode_state(next_state, next_launch)
                
                self.replay_memory.append(
                    new_state, action[0]-1, norm_land, norm_move, 
                    reward, new_next_state, done
                )

                rally_reward += reward
                state = next_state
                
                # update network
                self._update()
                self.total_timestamp += 1


            # rally end
            self.writer.add_scalar("Train/Rally Reward", rally_reward, rally_idx)
            self.writer.add_scalar("Train/Rally Round", info["round"][-1]-1, rally_idx)
            self.writer.add_scalar("Train/Step Epsilon", self.epsilon, self.total_timestamp)

            self.evaluate()
            self.total_rally += 1

    def evaluate(self):
        # do not evaluate
        if self.total_rally % self.eval_interval != 0:
            return
        
        rally_rounds = []
        rally_rewards = [0 for _ in range(self.evaluate_rallies)]
        for rally in range(1, self.evaluate_rallies + 1):
            print("===============================================")
            print("rally :", rally)
            state, info, done, launch = self.eval_env.reset()
            while not done :
                
                new_state = self._encode_state(state, launch)
                action_type, land, move, action_prob = self.act(new_state, launch, eval=True)

                action = (action_type, state[-1], land, move, action_prob)
                
                state, reward, info, done, launch = self.eval_env.step(action, launch)
                rally_rewards[rally-1] += reward

                if info["round"][-1] >= 56: # Too long
                    break

                print("state:", state)
                
            round = info['round'][-1]-1
            score = info['env_score']
            print("score: ", score)
            print("round: ", info['round'][-1]-1)
            rally_rounds.append(round)
        print("===============================================")

        self.writer.add_scalar("Evaluate/Max Round", max(rally_rounds), self.total_timestamp)
        self.writer.add_scalar("Evaluate/Min Round", min(rally_rounds), self.total_timestamp)
        self.writer.add_scalar("Evaluate/Average Round", np.mean(rally_rounds), self.total_timestamp)
        self.writer.add_scalar("Evaluate/Average Reward", np.mean(rally_rewards), self.total_timestamp)
        self._save(os.path.join(self.log_dir, f"model_{self.total_timestamp}_{int(np.mean(rally_rewards)*100)}.pth"))

    def _encode_state(self, state, launch):
        """
        Return the list format of state -> state and launch
        """
        action_type, ball, my_pos, opp_pos = state
        if state[0] is None:
            action_type = 11

        # new state
        norm_my_pos, norm_opp_pos, norm_ball, _, _ = self.encode_positions(my_pos, opp_pos, ball, (0,0), (0,0))
        state = [action_type-1] + list(norm_ball) + list(norm_my_pos) + list(norm_opp_pos) + [1 if launch else 0]
        return state

    def _update(self):
        
        if self.total_timestamp < self.warmup_steps:
            return
        
        self.epsilon = self.epsilon_final
        
        if self.total_timestamp % self.update_behavior_freq == 0:
            self._update_behavior()

        if self.total_timestamp % self.update_target_freq == 0:
            self._update_target()

    def _update_behavior(self):

        state, action_type, action_param, reward, next_state, done = self.replay_memory.sample(self.batch_size)

        batch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        batch_action_type = torch.tensor(action_type, dtype=torch.int64).to(self.device)
        batch_action_param = torch.tensor(action_param, dtype=torch.float).to(self.device)
        batch_reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        batch_next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        batch_done = torch.tensor(done, dtype=torch.int64).to(self.device)

        with torch.no_grad():
            next_action_param = self.target_param_net(batch_next_state)

            # DDQN
            pred_q = self.behavior_q_net(batch_next_state, next_action_param)
            _, next_action = pred_q.max(dim=1, keepdim=True)
            next_q = self.target_q_net(batch_next_state, next_action_param).gather(1, next_action)

            # Compute the TD error
            target_q = batch_reward + (1 - batch_done) * self.gamma * next_q

        # Compute current Q-values using policy network
        q_value = self.behavior_q_net(batch_state, batch_action_param)
        pred_q = q_value.gather(1, batch_action_type.view(-1, 1))
        
        loss_q = F.mse_loss(pred_q, target_q)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_q_net.parameters(), max_norm=self.max_grad_norm)
        self.q_optimizer.step()

        # ---------------------- optimize actor ----------------------
        action_param = self.behavior_param_net(batch_state)
        q_value = self.behavior_q_net(batch_state, action_param)

        #q_weights = F.softmax(q_value, dim=1)
        #loss_param = -(q_weights * q_value).sum(dim=1).mean()
        loss_param = -q_value.mean()
        
        self.param_optimizer.zero_grad()
        loss_param.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_param_net.parameters(), max_norm=self.max_grad_norm)
        self.param_optimizer.step()

        self.writer.add_scalar("Train/Loss Q", loss_q.item(), self.total_timestamp)
        self.writer.add_scalar("Train/Loss Param", loss_param.item(), self.total_timestamp)

    
    def _update_target(self):
        soft_update_target_network(self.behavior_q_net, self.target_q_net, self.tau_param)
        soft_update_target_network(self.behavior_param_net, self.target_param_net, self.tau_param)

    def _save(self, save_path):
        torch.save(
            {
                'q_net':        self.behavior_q_net.state_dict(),
                'param_net':    self.behavior_param_net.state_dict(),
            }, save_path)
        
    # load model
    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.behavior_q_net.load_state_dict(checkpoint['q_net'])
        self.behavior_param_net.load_state_dict(checkpoint['param_net'])

