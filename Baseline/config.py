import torch


class BaseParameters:
    seed = 42
    eval_interval = 1000
    evaluate_rallies = 10
    train_env = None
    eval_env = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent_name = "P-DQN"
    log_dir = "log/P-DQN_fix_epsilon_constrain"

class PDQNHyperParameters:
    num_episodes = 50000
    epsilon_final = 0.05
    replay_memory_size = 10000
    batch_size = 4096
    gamma = 1.0
    lr_q = 1e-4
    lr_param = 1e-4
    critic_hidden_layers = (64, 32)
    actor_hidden_layers = (64, 32)
    embed_dim = 4
    warmup_steps = 10000
    update_behavior_freq = 1
    update_target_freq = 1
    tau_param = 0.005
    sigma = 0.06
    max_grad_norm = 0.5

    # action_type, my_x, my_y, opp_x, opp_y, ball_x, ball_y, launch
    state_dim = 8
    action_dim = 10
    action_param_dim = 4

class HPPOHyperParameters:
    num_episodes = 100000
    horizon = 20
    update_sample_count = 10000
    batch_size = 4096
    gamma = 0.98
    gae_lambda = 0.95
    lr_actor = 1e-4
    lr_value = 1e-4
    critic_hidden_layers = (128, 64, 32)
    actor_hidden_layers = (128, 64, 32)
    update_epoch = 3
    clip_epsilon = 0.2
    max_gradient_norm = 0.5

    value_coefficient = 0.5
    entropy_coefficient = 0.01
    embed_dim = 4

    # action_type, my_x, my_y, opp_x, opp_y, ball_x, ball_y, launch
    state_dim = 8
    action_dim = 10
    action_param_dim = 4