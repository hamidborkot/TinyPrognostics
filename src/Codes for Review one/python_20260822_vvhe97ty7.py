import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/working/master_results_fixed.csv')

# Convert numeric columns safely
for col in ['rmse', 'mae', 'score', 'cz_rmse', 'acc', 'f1']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

def fmt(mean, std):
    if pd.isna(mean): return "—"
    return f"{mean:.2f} ± {std:.2f}"

print("="*70)
print("TABLE 1: C-MAPSS MAIN RESULTS (Multi-Task NanoSentry vs Baselines)")
print("="*70)
cmapss = df[df['dataset'] == 'C-MAPSS']
# Get NanoSentry Multi-task
ns = cmapss[(cmapss['model'] == 'NanoSentry') & (cmapss['variant'] == 'multi_task')]
# Get Baselines (single_task)
base = cmapss[cmapss['variant'] == 'single_task']

for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    print(f"\n--- {fd} ---")
    for model_name in ['NanoSentry', 'LSTM-64', 'CNN-32', 'Ridge']:
        if model_name == 'NanoSentry':
            subset_df = ns[ns['subset'] == fd]
        else:
            subset_df = base[(base['subset'] == fd) & (base['model'] == model_name)]
            
        rmse = fmt(subset_df['rmse'].mean(), subset_df['rmse'].std())
        mae = fmt(subset_df['mae'].mean(), subset_df['mae'].std())
        score = fmt(subset_df['score'].mean(), subset_df['score'].std())
        cz = fmt(subset_df['cz_rmse'].mean(), subset_df['cz_rmse'].std())
        print(f"{model_name:<12} | RMSE: {rmse:<15} | MAE: {mae:<15} | Score: {score:<15} | Crit RMSE: {cz}")

print("\n" + "="*70)
print("TABLE 2: CWRU RECORD-LEVEL FAULT DIAGNOSIS (10-Class)")
print("="*70)
cwru = df[df['dataset'] == 'CWRU']
for model_name in ['NanoSentry', 'CNN-32', 'LSTM-64']:
    subset_df = cwru[cwru['model'] == model_name]
    acc = fmt(subset_df['acc'].mean(), subset_df['acc'].std())
    f1 = fmt(subset_df['f1'].mean(), subset_df['f1'].std())
    print(f"{model_name:<12} | Accuracy: {acc:<15} | Macro-F1: {f1}")

print("\n" + "="*70)
print("TABLE 3: BATTERY LOCO (Aggregated across all 17 cells)")
print("="*70)
bat = df[df['dataset'] == 'Battery']
for model_name in ['NanoSentry', 'LSTM-64', 'CNN-32', 'Ridge']:
    if model_name == 'NanoSentry':
        subset_df = bat[(bat['model'] == 'NanoSentry') & (bat['variant'] == 'multi_task')]
    else:
        subset_df = bat[(bat['model'] == model_name) & (bat['variant'] == 'single_task')]
        
    rmse = fmt(subset_df['rmse'].mean(), subset_df['rmse'].std())
    mae = fmt(subset_df['mae'].mean(), subset_df['mae'].std())
    print(f"{model_name:<12} | RMSE: {rmse:<15} | MAE: {mae}")
