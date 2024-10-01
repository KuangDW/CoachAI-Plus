import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import uuid
import os

# 使用 ggplot 風格
style.use('ggplot')

def Plot_pie_chart(tactic_dict, player_name, dest_folder = './Tactic_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)
    
    """
    繪製圓餅圖
    """
    # 過濾掉數量為 0 的類別 以及 層次二三四
    tactic_dict = {k: v for k, v in tactic_dict.items() 
               if v > 0 and k not in ['Forehand_Lock', 'Backhand_Lock', 'FrontCourt_Lock', 'BackCourt_Lock', 'Four_Corners_Clear_Drop']}
    # print(tactic_dict)
    # tactic_dict = {k: v for k, v in tactic_dict.items() if v > 0}
    labels = tactic_dict.keys()
    sizes = tactic_dict.values()

    # 設置圓餅圖樣式
    colors = ['#f1f2eb', '#bfc1b8', '#d8dad3', '#a4c2a5', '#566246', '#4a4a48']
    plt.figure(figsize=(15, 10))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
            shadow=True, wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 14})

    plt.axis('equal')
    plt.title(f'Tactic Distribution of {player_name}', fontsize=24)

    id = uuid.uuid4()
    filename = f'{dest_folder}/{id}'
    plt.savefig(filename)
    plt.clf()
    plt.close()

    return id


def Plot_histogram(data_list, player1, player2, dest_folder = './Tactic_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)

    """
    比較直方圖
    """
    for dict1, playerA in data_list:
        for dict2, playerB in data_list:
            if playerA == player1 and playerB == player2:
                    categories = ['FCP', 'DC', 'FC', 'FhL', 'BhL', 'FcL', 'BcL', 'FCCD', 'No']
    
                    values1 = list(dict1.values())
                    values2 = list(dict2.values())
                    
                    # 設定條狀圖的位置
                    x = np.arange(len(categories))
                    width = 0.35  # 條狀的寬度
                    plt.figure(figsize=(10, 6))
                    
                    # 繪製兩組數據
                    plt.bar(x - width/2, values1, width, label=player1, color='#66b3ff', edgecolor='white')
                    plt.bar(x + width/2, values2, width, label=player2, color='#ff9999', edgecolor='white')
                    plt.xlabel('Tactic Category', fontsize=14)
                    plt.ylabel('Count', fontsize=14)
                    plt.title(f'Comparison of Tactics for {player1} and {player2}', fontsize=16)
                    plt.xticks(x, categories)
                    plt.legend()
                    id = uuid.uuid4()
                    filename = f'{dest_folder}/{id}'
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()



    # (dict1, player1), (dict2, player2) = data_list
    
    # 類別名稱，使用簡寫表示
    # #categories = [Full_Court_Pressure, Defensive_Counterattack, Four_Corner, Forehand_Lock, Backhand_Lock, FrontCourt_Lock, BackCourt_Lock, Four_Corners_Clear_Drop, No_tactic]
    # categories = ['FCP', 'DC', 'FC', 'FhL', 'BhL', 'FcL', 'BcL', 'FCCD', 'No']
    
    # values1 = list(dict1.values())
    # values2 = list(dict2.values())
    
    # # 設定條狀圖的位置
    # x = np.arange(len(categories))
    # width = 0.35  # 條狀的寬度
    # plt.figure(figsize=(10, 6))
    
    # # 繪製兩組數據
    # plt.bar(x - width/2, values1, width, label=player1, color='#66b3ff', edgecolor='white')
    # plt.bar(x + width/2, values2, width, label=player2, color='#ff9999', edgecolor='white')
    # plt.xlabel('Tactic Category', fontsize=14)
    # plt.ylabel('Count', fontsize=14)
    # plt.title(f'Comparison of Tactics for {player1} and {player2}', fontsize=16)
    # plt.xticks(x, categories)
    # plt.legend()
    # id = uuid.uuid4()
    # filename = f'{dest_folder}/{id}'
    # plt.savefig(filename)

    return id


def coord_diagram(tactic_record, win_record, player1, player2, dest_folder):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    for tacticA, playerA in tactic_record:
        if playerA == player1:
            for winA, playerC in win_record:
                if playerC == player1:
                    for tacticB, playerB in tactic_record:
                        if playerB == player2:
                            for winB, playerD in win_record:
                                if playerD == player2:
                                    # Calculate the total number of rallies for player A
                                    rally_num_A = (tacticA.get('Full_Court_Pressure', 0) + 
                                                tacticA.get('Four_Corner', 0) + 
                                                tacticA.get('Defensive_Counterattack', 0) +
                                                tacticA.get('No_tactic', 0))
                                    
                                    # Calculate the total number of rallies for player B
                                    rally_num_B = (tacticB.get('Full_Court_Pressure', 0) + 
                                                tacticB.get('Four_Corner', 0) + 
                                                tacticB.get('Defensive_Counterattack', 0) + 
                                                tacticB.get('No_tactic', 0))
                                    
                                    # Calculate usage for player A
                                    if rally_num_A > 0:
                                        usageA = {key: val / rally_num_A for key, val in tacticA.items()}
                                    else:
                                        usageA = {}

                                    # Calculate usage for player B
                                    if rally_num_B > 0:
                                        usageB = {key: val / rally_num_B for key, val in tacticB.items()}
                                    else:
                                        usageB = {}

                                    # Calculate Win Rates
                                    win_rate_A = {}
                                    win_rate_B = {}

                                    for tactic in ['Full_Court_Pressure', 'Defensive_Counterattack', 'Four_Corner', 'Forehand_Lock', 'Backhand_Lock', 'FrontCourt_Lock', 'BackCourt_Lock', 'Four_Corners_Clear_Drop', 'No_tactic']:
                                        # Win rate for player A
                                        wins_A = winA.get(tactic, 0)
                                        used_A = tacticA.get(tactic, 0)
                                        win_rate_A[tactic] = wins_A / used_A if used_A > 0 else 0

                                        # Win rate for player B
                                        wins_B = winB.get(tactic, 0)
                                        used_B = tacticB.get(tactic, 0)
                                        win_rate_B[tactic] = wins_B / used_B if used_B > 0 else 0

                                    # Get the tactics
                                    tactics = usageA.keys()

                                    # Create a new figure
                                    fig, ax = plt.subplots(figsize=(10, 6))

                                    # Plot for Player 1
                                    x1 = [usageA[tactic] for tactic in tactics]
                                    y1 = [win_rate_A[tactic] for tactic in tactics]
                                    ax.scatter(x1, y1, color='blue', label=player1)

                                    # Annotate Player 1 points
                                    for i, tactic in enumerate(tactics):
                                        ax.annotate(tactic, (x1[i], y1[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

                                    # Plot for Player 2
                                    x2 = [usageB[tactic] for tactic in tactics]
                                    y2 = [win_rate_B[tactic] for tactic in tactics]
                                    ax.scatter(x2, y2, color='red', label=player2)

                                    # Annotate Player 2 points
                                    for i, tactic in enumerate(tactics):
                                        ax.annotate(tactic, (x2[i], y2[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

                                    # Set labels and title
                                    ax.set_xlabel('Tactic Usage (%)', fontsize=12)
                                    ax.set_ylabel('Win Rate (%)', fontsize=12)
                                    ax.set_title(f'Tactic Usage vs. Win Rate for {player1} and {player2}', fontsize=16)

                                    # Display legend
                                    ax.legend(loc='best')

                                    # Add grid
                                    ax.grid(True)

                                    # Generate a unique ID and save the figure
                                    id = uuid.uuid4()
                                    filename = f'{dest_folder}/{id}.png'  # Ensure .png extension
                                    plt.savefig(filename)  # Save the figure

                                    # Clear and close the plot
                                    plt.clf()
                                    plt.close(fig)

                                    return id  # Return the filename

