import pandas as pd
import numpy as np
import gc

processed_data_path = '../data/processed'

def create_time_series_features():
    df = pd.read_parquet(f'{processed_data_path}/m5_lean_data.parquet')

    # sort by time
    df = df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)

    # create lag features
    grouped = df.groupby(['store_id','item_id'])['sales']

    #lag1: sales of yesterday
    #lag7: sales of the same day last week
    #lag28: sales of the same day last month
    lag_days = [1, 7, 28]

    for log in lag_days:
        df[f'sales_lag_{log}'] = grouped.shift(log)

    # create rolling features
    # prevent data leakage
    group_lag1 = df.groupby(['store_id','item_id'])['sales_lag_1']

    windows = [7,28]
    for w in windows:
        df[f'sales_roll_mean_{w}'] = group_lag1.transform(lambda x: x.rolling(w).mean())
        df[f'sales_roll_std_{w}'] = group_lag1.transform(lambda x: x.rolling(w).std())
        df[f'sales_roll_max_{w}'] = group_lag1.transform(lambda x: x.rolling(w).max())

    # convert categorical features to category dtype
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday', 'event_name_1', 'event_type_1']

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).astype('category')

    initial_rows = len(df)
    df = df.dropna(subset = ['sales_lag_28', 'sales_roll_mean_28', 'sell_price']).reset_index(drop=True)

    return df

if __name__ == '__main__':
    df_features = create_time_series_features()
    print('\nFinal feature dataset shape:', df_features.shape)
    save_path = f'{processed_data_path}/m5_features.parquet'
    df_features.to_parquet(save_path, index=False)
