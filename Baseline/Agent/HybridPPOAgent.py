import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from tqdm import tqdm


from Agent.BaseAgent import BaseAgent
from Net.Net import HybridActor, SimpleValueNet
# from ReplayBuffer.ReplayBuffer import ReplayMemory
from utils import hard_update_target_network, soft_update_target_network, get_prob_from_q


class HPPOAgent(BaseAgent):

    def __init__(self, base_param, hyper_param):
        super().__init__(base_param)
        ### parameter copying
        self.num_episodes = hyper_param.num_episodes
        self.horizon = hyper_param.horizon
        self.update_sample_count = hyper_param.update_sample_count
        self.batch_size = hyper_param.batch_size
        self.gamma = hyper_param.gamma
        self.gae_lambda = hyper_param.gae_lambda
        self.lr_actor = hyper_param.lr_actor
        self.lr_value = hyper_param.lr_value
        self.critic_hidden_layers = hyper_param.critic_hidden_layers
        self.actor_hidden_layers = hyper_param.actor_hidden_layers
        self.update_epoch = hyper_param.update_epoch
        self.clip_epsilon = hyper_param.clip_epsilon
        self.max_gradient_norm = hyper_param.max_gradient_norm
        self.value_coefficient = hyper_param.value_coefficient
        self.entropy_coefficient = hyper_param.entropy_coefficient
        self.state_dim = hyper_param.state_dim
        self.action_dim = hyper_param.action_dim
        self.action_param_dim = hyper_param.action_param_dim

        ### model construction
        self.critic_net = SimpleValueNet(base_param.state_dim, hyper_param.hidden_layers)
        self.actor_net = HybridActor(base_param.state_dim, base_param.action_dim, hyper_param.hidden_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.value_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_value)

        ### replay memory
        self.gae_replay_buffer = None

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
            state_value = self.critic_net(state_tensor)

            if launch == True:
                mask = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
            else:
                mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)

            discrete_logits, mean, logstd = self.actor_net(state_tensor)
            # mask out the invalid actions
            masked_logits = discrete_logits.clone()
            masked_logits[~mask] = -float('inf')

            discrete_dist = Categorical(logits=masked_logits)
            continous_dist = Normal(mean, logstd.exp())

            action_prob = discrete_dist.probs

            if not eval:
                ### sampling
                action_type = discrete_dist.sample()
                action_param = continous_dist.sample()
                
            else:
                ### greedy
                action_type = discrete_logits.argmax(dim=-1)
                action_param = mean
                
            # get log prob
            discrete_log_prob = discrete_dist.log_prob(action_type)
            continous_log_prob = continous_dist.log_prob(action_param)

            # make action_param to -1 ~ 1
            action_param = F.tanh(action_param)

        action_prob = action_prob.cpu().tolist()
        action_param = action_param[0].cpu().tolist()
        land, move = tuple(action_param[:2]), tuple(action_param[2:])
        
        # decode the action from [-1, 1] to correct range
        land, move = self.decode_positions(land, move)

        # action_type is 1 ~ 10
        # return more items -> state_value, logprob and entropy
        action = (action_type+1, land, move, action_prob)
        others = (state_value, discrete_log_prob, continous_log_prob, discrete_dist.entropy(), continous_dist.entropy())
        return action, others

    def train(self):

        for rally_idx in tqdm(range(self.num_episodes)):
            
            rally_reward = 0

            state, info, done, launch = self.train_env.reset()

            while not done:

                # reorganize a state with launch and get list format
                new_state = self._encode_state(state, launch)
                action, others = self.act(new_state, launch, eval=False)

                action_type, land, move, action_prob = action
                state_value, discrete_log_prob, continous_log_prob, discrete_entropy, continous_entropy = others
                
                action = (action_type, state[-1], land, move, action_prob)

                # step in
                next_state, reward, info, done, next_launch = self.train_env.step(action, launch)
                
                # normalize the positions
                _, _, _, norm_land, norm_move = self.encode_positions((0,0), (0,0), (0,0), land, move)

                if info["round"][-1] >= 58: # Too long
                    break
                if done:
                    next_state = state

                # reorganize a state with launch and get list format
                new_next_state = self._encode_state(next_state, next_launch)
                
                ### TODO ###
                self.replay_memory.append(
                    new_state, action[0]-1, norm_land, norm_move, 
                    reward, new_next_state, done
                )
                ### TODO ###

                if len(self.gae_replay_buffer) > self.update_sample_count:
                    self._update()
                    self.gae_replay_buffer.clear()

                rally_reward += reward
                state = next_state
                self.total_timestamp += 1


            # rally end
            self.writer.add_scalar("Train/Rally Reward", rally_reward, rally_idx)
            self.writer.add_scalar("Train/Rally Round", info["round"][-1]-1, rally_idx)

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
                action, _ = self.act(new_state, launch, eval=True)
                action_type, land, move, action_prob = action

                action = (action_type, state[-1], land, move, action_prob)
                
                state, reward, info, done, launch = self.eval_env.step(action, launch)
                rally_rewards[rally-1] += reward

                if info["round"][-1] >= 58: # Too long
                    break
                
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
        total_surrogate_loss, total_value_loss, total_entropy, total_loss = 0, 0, 0, 0
        ### TODO ###
        # batch training
        ### TODO ###


        self.writer.add_scalar("Train/Loss Q", loss_q.item(), self.total_timestamp)
        self.writer.add_scalar("Train/Loss Param", loss_param.item(), self.total_timestamp)

    
    def _update_target(self):
        hard_update_target_network(self.behavior_q_net, self.target_q_net)
        soft_update_target_network(self.behavior_param_net, self.target_param_net, self.tau_param)

    def _save(self, save_path):
        torch.save(
            {
                'critic_net':   self.critic_net.state_dict(),
                'actor_net':    self.actor_net.state_dict(),
            }, save_path)
        
    # load model
    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.critic_net.load_state_dict(checkpoint['critic_net'])
        self.actor_net.load_state_dict(checkpoint['actor_net'])

