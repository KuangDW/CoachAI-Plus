import subprocess

# 定義要執行的 Python 
scripts = [
    {"script": "homographyAndCombine.py", "args": []}, # 按照 data 裡面的 match 和 homography，把資料合併成 data_tables/raw_dataset.csv
    {"script": "dataPreprocessing.py", "args": []},     # data_tables/raw_dataset.csv -> data_tables/RRD_train_0.csv（過濾 missing value，轉換主觀座標）
    {"script": "data_preprocessing/ProduceContinuousDataset.py", "args": []}, # data_tables/RRD_train_0.csv -> 增加連續型欄位 -> data_tables/first_process_continuous.csv
    {"script": "data_preprocessing/ProduceDiscreteDataset.py", "args": []}, # data_tables/first_process_continuous.csv -> 增加離散型欄位 -> data_tables/second_process_discrete.csv
    {"script": "data_preprocessing/PlayerDataSeperation.py", "args": []}, # data_tables/second_process_discrete.csv -> 調整球種和分數，產生球員視角資料（資料集會變大） -> data_tables/all_dataset.csv
    {"script": "data_preprocessing/PCCT.py", "args": []}, # data_tables/all_dataset.csv --> data_tables/total_traj_train_0.pkl (需要的欄位變成 training dataset)
    {"script": "data_preprocessing/OPCCT.py", "args": []} # data_tables/total_traj_train_0.pkl 是混在一起的 rally info -> 拆成 player_train_0.pkl 和 opponent_train_0.pkl
]

# 依次執行程式
for script_info in scripts:
    script = script_info["script"]
    args = script_info["args"]
    # 創建命令
    command = ["python", script] + args
    try:
        # 執行腳本並等待完成
        result = subprocess.run(command, check=True)
        print(f"成功執行 {script}")
    except subprocess.CalledProcessError as e:
        print(f"執行 {script} 時發生錯誤：{e}")

