import pandas as pd

data = pd.read_csv("/data/users/sargent/dvfs_thesis/result-processing/task-frequency-statistics.csv")

data = data[data["bench"] == "lu_mustard"]
data = data[data["freq_mhz"] == 2040]
data = data[data["pe"] == 0]
data = data[data["task_id"] < 11]

data = data.sort_values("wait_time_ns")
print(data.head(30))