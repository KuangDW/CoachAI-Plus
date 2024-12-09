# average comparisons
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(
    df_1,
    df_2,
    target_player,
    dest_folder="./Shot_Evaluation/Result",
):
    match_ids = [df_1.iloc[0]["match_id"], df_2.iloc[0]["match_id"]]

    dataset = pd.concat([df_1, df_2], axis=0, ignore_index=True)

    dataset = dataset[
        (dataset["type"] != "Serve short") & (dataset["type"] != "Serve long")
    ]

    model = pd.read_csv(f"../win_prob_data/dataset_{target_player}.csv")
    dataset = pd.merge(
        dataset,
        model[["match_id", "set", "rally", "ball_round", "win_prob"]],
        on=["match_id", "set", "rally", "ball_round"],
        how="left",
    )

    # Calculate delta values for paces and win probabilities
    delta_paces = []
    delta_win_probs = []

    for _, rally_data in dataset.groupby(["match_id", "set", "rally"]):
        rally_data = rally_data.reset_index(drop=True)
        delta_pace = (
            rally_data["pace"].diff().fillna(rally_data["pace"])
        )  # First diff for pace
        delta_win_prob = (
            rally_data["win_prob"].diff().fillna(rally_data["win_prob"])
        )  # First diff for win_prob

        delta_paces.extend(delta_pace)
        delta_win_probs.extend(delta_win_prob)

    dataset["delta_pace"] = delta_paces
    dataset["delta_win_prob"] = delta_win_probs

    dataset = dataset[dataset["player"] == target_player]

    # Calculate averages for each shot type
    grouped_data = (
        dataset.groupby(["match_id", "type"])
        .agg(
            avg_pace=("pace", "mean"),
            avg_win_prob=("win_prob", "mean"),
            avg_delta_pace=("delta_pace", "mean"),
            avg_delta_win_prob=("delta_win_prob", "mean"),
        )
        .reset_index()
    )
    # print(grouped_data)

    # Extract unique shot types and sets for plotting
    shot_types = grouped_data["type"].unique()

    # Width for the bars
    bar_width = 0.35

    # Positions of the bars on the x-axis
    indices = np.arange(len(shot_types))

    # Prepare the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot avg_win_prob and avg_delta_win_prob for each set
    for i, match_id in enumerate(match_ids):
        data_for_set = grouped_data[grouped_data["match_id"] == match_id]
        ax1.bar(
            indices + i * bar_width,
            data_for_set["avg_win_prob"],
            bar_width,
            label=f"Match {match_id} Avg Win Prob",
        )
        # ax1.bar(indices + i * bar_width, data_for_set['avg_delta_win_prob'], bar_width, alpha=0.5, label=f'Set {s} Delta Win Prob')

    # Configure the first plot (avg_win_prob and avg_delta_win_prob)
    ax1.set_xlabel("Shot Type")
    ax1.set_ylabel("Win Probability")
    ax1.set_title("Average Win Probability by Shot Type")
    ax1.set_xticks(indices + bar_width / 2)
    ax1.set_xticklabels(shot_types, rotation=45)
    ax1.legend()

    # Plot avg_pace and avg_delta_pace for each set
    for i, match_id in enumerate(match_ids):
        data_for_set = grouped_data[grouped_data["match_id"] == match_id]
        # ax2.bar(indices + i * bar_width, data_for_set['avg_pace'], bar_width, label=f'Set {s} Avg Pace')
        ax2.bar(
            indices + i * bar_width,
            data_for_set["avg_delta_pace"],
            bar_width,
            label=f"Match {match_id} Avg Delta Pace",
        )

    # Configure the second plot (avg_pace and avg_delta_pace)
    ax2.set_xlabel("Shot Type")
    ax2.set_ylabel("Pace")
    ax2.set_title("Average Delta Pace by Shot Type")
    ax2.set_xticks(indices + bar_width / 2)
    ax2.set_xticklabels(shot_types, rotation=45)
    ax2.legend()

    # Adjust layout and display the plot
    plt.tight_layout()

    # Show the plot
    id = uuid.uuid4()
    filename = f"{dest_folder}/{target_player}_shots_comparison{id}.png"
    plt.savefig(filename)
    plt.close()

    return id
