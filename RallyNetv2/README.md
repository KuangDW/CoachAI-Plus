# RallyNetv2: Official PyTorch Implementation
This is the official PyTorch implementation of RallyNetv2, an improvement over RallyNet: Offline Imitation of Turn-Based Player Behavior via Experiential Contexts and Brownian Motion (ECML PKDD 2024) [[1]](https://arxiv.org/abs/2403.12406), validated on the largest publicly available badminton dataset: ShuttleSet (KDD'23) [[2]](https://arxiv.org/abs/2306.04948). Our official implementations of Coach AI Badminton Project Page: [this https URL](https://github.com/wywyWang/CoachAI-Projects)

## Overview

RallyNet was the first imitation learning model specifically designed for badminton. It proposed using player experiences (action sequences) to construct intents and linked these intents with interaction models using Latent Geometric Brownian Motion (LGBM). This framework aimed to provide a more interpretable representation of player interactions for sports analytics.

RallyNet requires both agents to select accurate intents, which are then fed into LGBM to generate the correct interaction sequences. However, intent prediction is often noisy. To address this, we propose a simpler method to directly learn the correct interaction sequences. RallyNetv2 extends the interaction length of LGBM, enabling it to directly map to the correct mixed-action sequences. We found that this method aligns better with the true action sequences.

In the rally content reconstruction experiment, we achieved significant improvements in three metrics: Drop Location Error (DTW Distance) of 0.3695, Shot Type Error (CTC Loss) of 8.9367, and Movement Error (DTW Distance) of 0.2380, representing improvements of 41%, 50%, and 31%, respectively, over the original RallyNet.

| Model        | Landing Location (DTW) | Shot Type Distribution (CTC) | Moving Location (DTW) |
|--------------|------------------------|-----------------------------|-----------------------|
| RallyNet     | 0.6263                 | 17.9273                     | 0.3446                |
| RallyNetv2   | 0.3695                 | 8.9367                      | 0.2380                |

## Set up
1. Create the environment
```
conda env create -f environment.yml
```

2. Data Preprocessing 
Our dataset is based on the largest public badminton dataset, ShuttleSet [2]. Place the dataset in the `Current_dataset/data/` folder. For detailed dataset structure, refer to ShuttleSet's documentation. To generate the training data for RallyNetv2, run the following command in the `Current_dataset` directory:
```
python run.py
```
This will generate `player_train_0.pkl` and `opponent_train_0.pkl`, which contain the trajectories (state-action pairs) of the rally initiator and the receiver, respectively. The `target_players_ids` file lists the IDs of different players.

> For the Evaluation Modules provided by CoachAI+ Badminton Environment, the required match data format is the `all_dataset.csv` file located in the `Current_dataset/data_tables/` directory.

## Training & Evaluation
To train an agent using RallyNetv2, follow these steps:
1. Train the inverse dynamic model:
```
python INV.py 
```
2. Train RallyNetv2:
```
python RallyNetv2.py 
```
Trained models will be saved in the `dump` folder. For evaluation, set the `eval` parameter in the `main` function to True. The `split_ratio` parameter can be used to control the ratio of training to evaluation data.

## Behavior Cloning Model
To train a Behavior Cloning model, run:
```
python BC_train.py 
```
The model weights will be saved as `BC_weight.pth`. You can then copy this file into the `BadmintonEnv/Agent` directory for use.

## Citation
If you use our dataset or find our work is relevant to your research, please cite:
```
@inproceedings{DBLP:conf/pkdd/WangWHP24,
  author       = {Kuang{-}Da Wang and
                  Wei{-}Yao Wang and
                  Ping{-}Chun Hsieh and
                  Wen{-}Chih Peng},
  title        = {Offline Imitation of Badminton Player Behavior via Experiential Contexts
                  and Brownian Motion},
  booktitle    = {{ECML/PKDD} {(10)}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14950},
  pages        = {348--364},
  publisher    = {Springer},
  year         = {2024}
}
```

## References
[1] Want et al. "Offline Imitation of Badminton Player Behavior via Experiential Contexts and Brownian Motion." ECML PKDD'24.

[2] Wang et al. "ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis." KDD'23