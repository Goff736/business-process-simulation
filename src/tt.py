from pathlib import Path
import pandas as pd
import pm4py

# input (your current file)
file_path = r"C:\Users\SBS\Downloads\1st ex PRAK\BPI Challenge 2017_1_all\BPI Challenge 2017.xes.gz"

# output (must be inside this repo)
repo_root = Path(r"C:\Users\SBS\repo\Business-Process-Optimization-PART2")
out_csv = repo_root / "data" / "bpi2017.csv"

log = pm4py.read_xes(file_path)
df = pm4py.convert_to_dataframe(log)

# make sure timestamps parse correctly
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")

df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)