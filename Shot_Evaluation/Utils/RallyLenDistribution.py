import os
import pandas as pd
import matplotlib.pyplot as plt

def rally_len_distribution(dict1):
    os.makedirs('./Shot_Evaluation/Result', exist_ok=True)
    lengths_df1 = pd.DataFrame(list(dict1.items()), columns=['Length', 'Count'])

    plt.figure(figsize=(12, 6))
    plt.bar(lengths_df1['Length'] - 0.2, lengths_df1['Count'], width=0.4, color='red', label='Real')
    plt.xlabel('Length of Rally')
    plt.ylabel('Count')
    plt.title('Distribution of Rally Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Shot_Evaluation/Result/rallyLength.png')