# stats.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os

def Player_ShotType_histogram(df, player_name, dest_folder = './Visualization_And_Win_Lose_Reason_Statistics/Result'):
    os.makedirs(dest_folder, exist_ok=True)

    type_dict = {1: 'Serve short', 2: 'Clear', 3: 'Push Shot', 4: 'Smash', 5: 'Smash Defence', 6: 'Drive',
                 7: 'Net Shot', 8: 'Lob', 9: 'Drop', 10: 'Serve long', 11: 'Missed shot'}
    
    player_data = df[df['player'] == player_name]
    player_data['type'] = player_data['type'].map(type_dict)

    # 統計每個 type 的數量
    type_counts = player_data['type'].value_counts()
    type_probabilities = type_counts / type_counts.sum()

    # 繪製機率直方圖
    plt.figure(figsize=(10, 6))
    sns.barplot(x=type_probabilities.index, y=type_probabilities.values, palette='coolwarm')
    plt.title(f"Type Probability Distribution for {player_name}", fontsize=16)
    plt.xlabel('Shot Type', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    sns.despine()  # 去掉多餘的圖框線條
    plt.xticks(rotation=45)
    plt.tight_layout()

    id = uuid.uuid4()
    filename = f'{dest_folder}/{id}.png'
    plt.savefig(filename)

    return id