import pandas as pd

df = pd.read_csv("/kaggle/working/polished_results.csv")

for c in ["rmse", "mae", "score", "cz_rmse", "acc", "f1"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("=" * 80)
print("ATTENTION COMPARISON BY SUBSET")
print("=" * 80)

att = df[
    (df["dataset"] == "C-MAPSS") &
    (df["model"].str.startswith("NanoSentry_")) &
    (df["variant"] == "multi_task")
]

att_summary = (
    att.groupby(["subset", "model"])
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        cz_mean=("cz_rmse", "mean"),
        cz_std=("cz_rmse", "std"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std")
    )
    .reset_index()
)

print(att_summary)

print("\n" + "=" * 80)
print("ABLATION BY SUBSET")
print("=" * 80)

abl = df[
    (df["dataset"] == "C-MAPSS") &
    (df["model"] == "NanoSentry_Ablation")
]

abl_summary = (
    abl.groupby(["subset", "variant"])
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        cz_mean=("cz_rmse", "mean"),
        cz_std=("cz_rmse", "std")
    )
    .reset_index()
)

print(abl_summary)