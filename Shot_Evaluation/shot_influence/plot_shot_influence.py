# store win_prob and attention weights between rallies to datset
import datetime
import json
import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shot_influence.rally_classifier as rc
import shot_influence.train as train
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# exist models
target_players = [
    "Kento MOMOTA",
    "CHOU Tien Chen",
    "Anthony Sinisuka GINTING",
    "CHEN Long",
    "CHEN Yufei",
    "TAI Tzu Ying",
    "Viktor AXELSEN",
    "Anders ANTONSEN",
    "PUSARLA V. Sindhu",
    "WANG Tzu Wei",
    "Khosit PHETPRADAB",
    "Jonatan CHRISTIE",
    "NG Ka Long Angus",
    "SHI Yuqi",
    "Ratchanok INTANON",
    "An Se Young",
    "LEE Cheuk Yiu",
    "Michelle LI",
    "Carolina MARIN",
]


def preprocess(dataset, target):
    player_list = dataset["player"].unique()
    opponent = player_list[0] if player_list[1] == target else player_list[1]
    # update round score
    for rally_id in dataset["rally_id"].unique():
        rally_mask = dataset["rally_id"] == rally_id
        getpoint_player = dataset[rally_mask]["getpoint_player"].iloc[-1]
        if getpoint_player == "player":
            dataset.loc[rally_mask, "player_score"] -= 1
        else:
            dataset.loc[rally_mask, "opponent_score"] -= 1

    # get point player
    # Function to update getpoint_player to real names
    def update_getpoint_player(row):
        getpoint_player = row["getpoint_player"]
        if getpoint_player == "player":
            return target
        elif getpoint_player == "opponent":
            return opponent
        else:
            return None

    # Update the getpoint_player column with real names
    dataset["getpoint_player"] = dataset.apply(update_getpoint_player, axis=1)

    # Propagate the getpoint_player value to all rows in the same rally
    dataset["getpoint_player"] = dataset.groupby(["match_id", "rally_id"])[
        "getpoint_player"
    ].transform("last")

    # add time proportion
    last_time = dataset.loc[
        dataset.groupby(["match_id", "set", "rally"])["ball_round"].idxmax()
    ]
    start_time = dataset.loc[
        dataset.groupby(["match_id", "set", "rally"])["ball_round"].idxmin()
    ]

    time_proportion = []
    for id, row in dataset.iterrows():
        last_rally_time = last_time.loc[
            (last_time["match_id"] == row["match_id"])
            & (last_time["set"] == row["set"])
            & (last_time["rally"] == row["rally"]),
            "time",
        ].values[0]
        start_rally_time = start_time.loc[
            (start_time["match_id"] == row["match_id"])
            & (start_time["set"] == row["set"])
            & (start_time["rally"] == row["rally"]),
            "time",
        ].values[0]
        print()
        start_rally_time = (
            datetime.datetime.strptime(start_rally_time.split()[0], "%H:%M:%S")
            - datetime.datetime(1900, 1, 1)
        ).total_seconds()
        last_rally_time = (
            datetime.datetime.strptime(last_rally_time.split()[0], "%H:%M:%S")
            - datetime.datetime(1900, 1, 1)
        ).total_seconds()
        current_rally_time = (
            datetime.datetime.strptime(row["time"].split()[0], "%H:%M:%S")
            - datetime.datetime(1900, 1, 1)
        ).total_seconds()
        if start_rally_time == last_rally_time:
            time_proportion.append(1)
        else:
            time_proportion.append(
                (current_rally_time - start_rally_time)
                / (last_rally_time - start_rally_time)
            )
    dataset["time_proportion"] = time_proportion
    dataset = dataset.drop(dataset.columns[0], axis=1)

    # Deal with areas... todo
    def remap_grid_area(old_area):
        # Define the new groupings
        if old_area in [20, 24]:
            return 1
        elif old_area in [17, 21]:
            return 2
        elif old_area in [4]:
            return 3
        elif old_area in [1]:
            return 4
        elif old_area in [8, 12, 16]:
            return 5
        elif old_area in [5, 9, 13]:
            return 6
        elif old_area in [18, 19, 22, 23]:
            return 7
        elif old_area in [6, 7, 10, 11, 14, 15]:
            return 8
        elif old_area in [2, 3]:
            return 9
        # Continue mapping as needed
        else:
            return -1

    dataset["hit_area"] = dataset["hit_area"].apply(remap_grid_area)
    dataset["player_location_area"] = dataset["player_location_area"].apply(
        remap_grid_area
    )
    dataset["opponent_location_area"] = dataset["opponent_location_area"].apply(
        remap_grid_area
    )

    # Deal with backhand and aroundhead
    dataset["backhand"] = dataset["backhand"].fillna(0).astype(int)
    dataset["aroundhead"] = dataset["aroundhead"].fillna(0).astype(int)

    dataset["is_target_turn"] = (dataset["player"] == target).astype(int)
    dataset["is_target_win"] = (dataset["getpoint_player"] == target).astype(int)

    dataset["roundscore_diff"] = dataset["player_score"] - dataset["opponent_score"]

    continuous_score_all = None
    getpoint_player = dataset.groupby(["match_id", "set", "rally_id"])[
        "getpoint_player"
    ].last()
    score_player_change = (
        getpoint_player != getpoint_player.groupby(level=[0, 1]).shift()
    )
    continuous_score = (
        (score_player_change.groupby(score_player_change.cumsum()).cumcount() + 1)
        .groupby(level=[0, 1])
        .shift(fill_value=0)
        .rename("continuous_score")
    )
    continuous_score = continuous_score.where(
        getpoint_player.groupby(level=[0, 1]).shift() == target, -continuous_score
    )
    if continuous_score_all is None:
        continuous_score_all = continuous_score
    else:
        continuous_score_all = pd.concat([continuous_score_all, continuous_score])

    dataset = dataset.join(continuous_score_all, on=["match_id", "set", "rally_id"])

    return dataset


def run_model(dataset, target_player):
    encode_columns = []
    shot_predictors = ["is_target_turn", "aroundhead", "backhand", "time_proportion"]
    rally_predictors = ["roundscore_diff", "continuous_score"]
    target = "is_target_win"

    seq_len = dataset.groupby("rally_id").size().max()
    seq_len += 1 if seq_len % 2 == 1 else 2

    encoded = pd.get_dummies(dataset, columns=encode_columns)
    # different target models have different type encoding
    with open("./model/" + target_player + "type_mapping.json", "r") as f:
        mapping = json.load(f)
        uniques_type = mapping["uniques_type"]

    type_to_code = {t: i for i, t in enumerate(uniques_type)}
    encoded["type"] = encoded["type"].map(type_to_code).astype(int) + 1

    encoded["hit_area"] = encoded["hit_area"] + 1
    encoded["player_location_area"] = encoded["player_location_area"] + 1
    encoded["opponent_location_area"] = encoded["opponent_location_area"] + 1
    shot_predictors = [
        c
        for c in encoded.columns
        if any(c.startswith(f"{p}_") for p in shot_predictors) or c in shot_predictors
    ]

    test_data = encoded

    (test_shots, test_shot_types), (test_rallies, test_target, test_rally_id) = (
        train.prepare_data(
            test_data,
            [
                shot_predictors,
                ["hit_area", "player_location_area", "opponent_location_area", "type"],
            ],
            [rally_predictors, target, "rally_id"],
            pad_to=seq_len,
        )
    )
    seq_len = test_shots.shape[1]

    test_hit_area_encoded = test_shot_types[:, :, 0].copy()
    test_player_area_encoded = test_shot_types[:, :, 1].copy()
    test_opponent_area_encoded = test_shot_types[:, :, 2].copy()
    test_shot_types = test_shot_types[:, :, 3].copy()
    test_time_proportion = test_shots[:, :, 2].copy()  # time proportion
    test_shots = np.delete(test_shots, 2, axis=2)

    shot_predictors.remove("time_proportion")

    regularizer = tf.keras.regularizers.l2(0.01)
    optimizer = "adam"
    loss = "binary_crossentropy"
    metrics = ["AUC", "binary_accuracy"]
    epochs = 100

    n_shot_types = len(uniques_type) + 1
    n_area_types = encoded["player_location_area"].nunique() + 1
    cnn_kwargs = {
        "filters": 32,
        "kernel_size": 3,
        "kernel_regularizer": regularizer,
        "activation": "relu",
    }
    rnn_kwargs = {"units": 32, "kernel_regularizer": regularizer}
    dense_kwargs = {"kernel_regularizer": regularizer}

    batch_size = 32
    MODEL_NAME = target_player
    model_file = "./model/" + MODEL_NAME + "/transformer"

    # Avoid tensorflow use full memory
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    prediction_model, attention_model = rc.bad_net(
        (seq_len, len(shot_predictors)),
        embed_types_size=n_shot_types,
        embed_area_size=n_area_types,
        rally_info_shape=len(rally_predictors),
        cnn_kwargs=cnn_kwargs,
        rnn_kwargs=rnn_kwargs,
        dense_kwargs=dense_kwargs,
    )
    prediction_model.load_weights(model_file)

    test_x = [
        test_hit_area_encoded,
        test_player_area_encoded,
        test_opponent_area_encoded,
        test_shots,
        test_shot_types,
        test_time_proportion,
        test_rallies,
    ]
    prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # test_results = prediction_model.evaluate(test_x, test_target)
    y_pred = prediction_model.predict(test_x)

    def generate_subsequences(data):
        for i in range(1, len(data) + 1):
            yield data[:i]

    # Define the maximum sequence length
    seq_len = dataset.groupby("rally_id").size().max()
    seq_len += 1 if seq_len % 2 == 1 else 2

    # Get all unique rally IDs
    rally_ids = dataset["rally_id"].unique()

    # Create new columns for storing attention values
    for i in range(1, seq_len):  # Assuming maximum rally length is 52
        dataset[f"ball_round{i}"] = np.nan

    # Create the 'win_prob' column
    dataset["win_prob"] = np.nan

    # Define the predictors and target columns
    shot_predictors = ["aroundhead", "backhand", "is_target_turn", "time_proportion"]

    # Process each rally
    for rally_id in rally_ids:
        rally_mask = encoded["rally_id"] == rally_id
        rally_data = encoded[rally_mask]

        # Generate subsequences from the raw rally data
        subsequences = [
            rally_data.iloc[subsequence_indices]
            for subsequence_indices in generate_subsequences(range(len(rally_data)))
        ]

        for i, subsequence in enumerate(subsequences):
            (test_shots, test_shot_types), (
                test_rallies,
                test_target,
                test_rally_id,
            ) = train.prepare_data(
                subsequence,
                [
                    shot_predictors,
                    [
                        "hit_area",
                        "player_location_area",
                        "opponent_location_area",
                        "type",
                    ],
                ],
                [rally_predictors, target, "rally_id"],
                pad_to=seq_len,
            )

            test_hit_area_encoded = test_shot_types[:, :, 0].copy()
            test_player_area_encoded = test_shot_types[:, :, 1].copy()
            test_opponent_area_encoded = test_shot_types[:, :, 2].copy()
            test_shot_types = test_shot_types[:, :, 3].copy()
            test_time_proportion = test_shots[:, :, 2].copy()  # time proportion
            test_shots = np.delete(test_shots, 2, axis=2)

            test_x = [
                test_hit_area_encoded,
                test_player_area_encoded,
                test_opponent_area_encoded,
                test_shots,
                test_shot_types,
                test_time_proportion,
                test_rallies,
            ]
            y_pred = prediction_model.predict(test_x)
            prediction = y_pred[0][0]
            rally_data.at[rally_data.index[i], "win_prob"] = prediction

            y_att = attention_model.predict(test_x)
            y_att = np.squeeze(y_att, axis=2)
            if y_att[0][-1][0] == 0.0:
                attention = y_att[0][-1][1 : len(subsequences) + 1]
            else:
                attention = y_att[0][-1][: len(subsequences)]

            # Store the attention values in the corresponding columns
            for j, att in enumerate(attention):
                rally_data.at[rally_data.index[i], f"ball_round{j+1}"] = att

        # Update the dataset with the new columns
        dataset.update(rally_data)

    dataset["type"] = dataset["type"].map(lambda x: uniques_type[int(x) - 1])
    # Save the updated dataset to a new CSV file
    return dataset


def plot_winprob(
    df_Real,
    df_model,
    match_id,
    set_num,
    rally,
    player_name,
    dest_folder="./Shot_Evaluation/Result",
):
    os.makedirs(dest_folder, exist_ok=True)
    df_Real = df_Real[
        (df_Real["match_id"] == match_id)
        & (df_Real["set"] == set_num)
        & (df_Real["rally"] == rally)
    ]
    paces = []
    for index, _ in df_Real.iterrows():
        pace = df_Real.loc[index, "pace"]
        if df_Real.loc[index, "player"] != player_name:
            pace = pace * (-1)
        if df_Real.loc[index, "hit_height"] != 1:
            pace = pace * (-1)
        paces.append(pace)

    win_probabilities = df_model[
        (df_model["match_id"] == match_id)
        & (df_model["set"] == set_num)
        & (df_model["rally"] == rally)
    ]["win_prob"].reset_index(drop=True)
    shot_types = df_Real["type"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Generate x-axis values (time steps)
    ball_rounds = list(range(1, len(win_probabilities) + 1))

    # Plot the line graph
    ax1.plot(
        ball_rounds,
        win_probabilities,
        marker="o",
        linestyle="-",
        label="Win Probability",
    )
    ax1.set_title(f"Win Probability During the Rally ({player_name})")
    ax1.set_xlabel("Ball Round")
    ax1.set_ylabel("Win Probability")
    ax1.grid(True)

    # Add bars to indicate changes in probability
    for i in range(1, len(win_probabilities)):
        ax1.bar(
            ball_rounds[i],
            win_probabilities[i] - win_probabilities[i - 1],
            color="gray",
            alpha=0.5,
            width=0.3,
            label="Delta Win Probability" if i == 1 else "",
        )
    ax1.legend()

    ax2.plot(
        ball_rounds,
        paces[: len(ball_rounds)],
        marker="x",
        linestyle="--",
        color="r",
        label="Pace",
    )
    for i, txt in enumerate(shot_types):
        ax2.text(
            ball_rounds[i], paces[i] - 0.05, txt, fontsize=9, ha="center", va="top"
        )
    ax2.set_title(f"Pace with Momentum During the Rally ({rally})")
    ax2.set_xlabel("Ball Round")
    ax2.set_ylabel("Pace")
    ax2.grid(True)

    for i in range(1, len(paces)):
        ax2.bar(
            ball_rounds[i],
            paces[i] - paces[i - 1],
            color="gray",
            alpha=0.5,
            width=0.3,
            label="Delta Pace" if i == 1 else "",
        )
    ax2.legend()
    plt.tight_layout()

    # Show the plot
    id = uuid.uuid4()
    # filename = f"{dest_folder}/{player_name}_{rally}.png"
    filename = f"{dest_folder}/{id}.png"
    plt.savefig(filename)
    plt.close()

    return id


def main(
    player_name,
    match_id,
    set_num,
    rally,
    df_real,
    df_model,
    dest_folder="../Shot_Evaluation/Result",
):

    filtered_df = df_model[
        (df_model["match_id"] == match_id)
        & (df_model["set"] == set_num)
        & (df_model["rally"] == rally)
    ]
    if filtered_df.empty:
        dataset = preprocess(df_real, player_name)
        df_model = run_model(dataset, player_name)
    id = plot_winprob(
        df_real, df_model, match_id, set_num, rally, player_name, dest_folder
    )

    return id
