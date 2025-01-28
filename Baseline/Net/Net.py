import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


# class ActionTypeEncoder(nn.Module):
#     def __init__(self, embed_dim, action_dim):
#         self.embedding_layer = nn.Embedding(action_dim, embed_dim, max_norm=1.0)
#         self.embed_dim = embed_dim
#         self.action_dim = action_dim

#     def forward(self, action_type):
#         return self.embedding_layer(action_type)

class SimpleDQN(nn.Module):

    def __init__(self, state_dim, action_param_dim, action_dim, hidden_layers):
        """
        :param state_dim:
        :param action_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_param_dim, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        # Value stream
        self.value_layer = nn.Linear(hidden_layers[-1], 1)

        # Advantage stream
        self.advantage_layer = nn.Linear(hidden_layers[-1], action_dim)

        init_weights(self.modules())

    def forward(self, state, action_params):

        # concatenate (state, continuous_action)
        x = torch.cat((state, action_params), dim=1)

        for i in range(len(self.layers)):
            # linear network
            x = self.layers[i](x)
            # non-linear mapping
            x = F.relu(x)

        value = self.value_layer(x)  # Output shape: [batch_size, 1]
        advantage = self.advantage_layer(x)  # Output shape: [batch_size, action_dim]

        # Combine Value and Advantage into Q-values
        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_value
    

class SimpleParamNet(nn.Module):

    def __init__(self, state_dim, action_parameter_dim, hidden_layers=(256, 128, 64)):
        """
        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.action_parameters_output_layer = nn.Linear(
            hidden_layers[-1], action_parameter_dim
        )

        self.action_parameter_dim = action_parameter_dim
        init_weights(self.modules())

    def forward(self, state):
        
        x = state
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.relu(x)

        action_param = self.action_parameters_output_layer(x)
        action_param = F.tanh(action_param)

        return action_param
    
class SimpleValueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers=(256, 128, 64)):
        """
        :param state_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.value_output_layer = nn.Linear(hidden_layers[-1], 1)

        init_weights(self.modules())


    def forward(self, state):
        
        x = state
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.relu(x)

        value = self.value_output_layer(x)

        return value
    
class HybridActor(nn.Module):

    def __init__(self, state_dim, action_dim, action_param_dim, hidden_layers):
        """
        :param state_dim:
        :param action_dim:
        :param action_param_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        # discrete action
        self.discrete_actor = nn.Linear(hidden_layers[-1], action_dim)
        self.mean_actor = nn.Linear(hidden_layers[-1], action_param_dim)
        self.logstd_actor = nn.Linear(hidden_layers[-1], action_param_dim)

        init_weights(self.modules())
    
    def forward(self, state):
        
        x = state
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.relu(x)

        discrete_logits = self.discrete_actor(x)
        mean = self.mean_actor(x)
        logstd = self.logstd_actor(x)

        return discrete_logits, mean, logstd