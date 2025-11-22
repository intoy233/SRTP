import pandas as pd

df = pd.read_csv('data/final_bridge_dataset.csv')
print('=== Value Ranges ===')
numeric_cols = ['Span_m', 'Width_m', 'Height_m', 'Natural_Freq_Hz',
                'Critical_Wind_Speed_ms', 'Max_Amplitude_mm', 'Damping_Ratio']
for col in numeric_cols:
    print(f'{col}:')
    print(f'  Min: {df[col].min():.4f}')
    print(f'  Max: {df[col].max():.4f}')
    print(f'  Mean: {df[col].mean():.4f}')
    print(f'  Median: {df[col].median():.4f}')
