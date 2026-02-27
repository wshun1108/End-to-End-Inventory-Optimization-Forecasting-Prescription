import pandas as pd
import numpy as np
import gc

row_data_path = '../data/raw/m5-forecasting-accuracy'
processsed_data_path = '../data/processed'

def create_features():

    # read row data
    df_sales = pd.read_csv(f'{row_data_path}/sales_train_evaluation.csv')

    # select target store and category
    target_store = 'CA_1'
    target_cat = 'HOBBIES'

    df_sales = df_sales[df_sales['store_id'] == target_store]
    df_sales = df_sales[df_sales['cat_id'] == target_cat]

    # invert the data from wide format to long format
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df_melted = pd.melt(df_sales, id_vars=id_vars, var_name='d', value_name='sales')

    del df_sales
    gc.collect()

    # merge with calendar data
    calender = pd.read_csv(f'{row_data_path}/calendar.csv')
    calender = calender[['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year', 'event_name_1', 'event_type_1', 'snap_CA']]

    df_merged = df_melted.merge(calender, on='d', how='left')
    del calender, df_melted
    gc.collect()

    # merge with price data
    prices = pd.read_csv(f'{row_data_path}/sell_prices.csv')
    prices = prices[prices['store_id'] == target_store]

    df_final = df_merged.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices, df_merged
    gc.collect()

    #optimize data types
    df_final['date'] = pd.to_datetime(df_final['date'])
    df_final['sales'] = df_final['sales'].astype(np.float32)

    return df_final

if __name__ == "__main__":
    df = create_features()
    save_path = f'{processsed_data_path}/m5_lean_data.parquet'
    df.to_parquet(save_path, index=False)
    print("Saved!")
