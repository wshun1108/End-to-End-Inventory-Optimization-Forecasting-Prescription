import pandas as pd
import numpy as np
import os

def inventory_backtest():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(os.path.dirname(current_dir), 'data', 'processed')

    df = pd.read_parquet(f'{processed_data_path}/m5_predictions.parquet')
    required_cols = ['item_id', 'date', 'sales', 'sell_price', 'pred_q50', 'pred_q90']
    missing_cols = [col for col in required_cols if col not in df.columns]

    df = df.sort_values(['item_id', 'date']).reset_index(drop=True)

    # set the parameters for the backtest
    # 25% gross profit margin
    # Round up the purchase quantity
    # annual holding cost is 25% of the purchase cost
    # Strategy A: Purchase based on median forecast (Purchase the amount expected to be sold)
    # Strategy B: Purchase goods based on the 90th percentile (Taking into account risks and safety stock)
    # The toy spoilage rate is low, and we assume it to be 0.01.
    df['cost'] = df['sell_price'] * 0.75
    daily_holding_cost = 0.25 / 365
    df['daily_holding_unit_cost'] = df['cost'] * daily_holding_cost
    daily_spoilage_rate = 0.01

    initial_len = len(df)
    df = df.dropna(subset=['sales', 'sell_price', 'pred_q50', 'pred_q90'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows containing NaN values.")

    # function of calculate profit
    def simulate_strategy(data, policy_col, strategy_name):
        total_profit = 0
        stockout_days = 0
        total_spoilage_cost = 0
        total_days = len(data)
        
        # 用于记录每日详情的列表
        daily_profits = []
        daily_inventories = []

        for item_id, group in data.groupby('item_id', observed=True):
            current_inventory = 0

            for idx, row in group.iterrows():
                # morning: receive new inventory
                target_level = np.ceil(row[policy_col])
                actual_demand = row['sales']

                order_qty = max(0,target_level - current_inventory)

                stock_available = current_inventory + order_qty

                # day: sell the stock
                sales_qty = min(stock_available, actual_demand)

                if actual_demand > stock_available:
                    stockout_days += 1

                # night: settlement
                unsold_inventory = stock_available - sales_qty
                spoilage_qty = unsold_inventory * daily_spoilage_rate

                current_inventory = unsold_inventory - spoilage_qty

                revenue = sales_qty * row['sell_price']
                cogs = sales_qty * row['cost']

                holding_cost = unsold_inventory * row['daily_holding_unit_cost']
                spoilage_cost = spoilage_qty * row['cost']

                daily_profit = revenue - cogs - holding_cost - spoilage_cost
                total_profit += daily_profit
                total_spoilage_cost += spoilage_cost

                daily_profits.append({'index': idx, f'profit_{strategy_name}': daily_profit})
                daily_inventories.append({'index': idx, f'inv_{strategy_name}': stock_available})
        
        if total_days > 0:
            service_level = (1 - (stockout_days / total_days)) * 100
        else:
            service_level = 0.0

        df_profit = pd.DataFrame(daily_profits).set_index('index')
        df_inv = pd.DataFrame(daily_inventories).set_index('index')

        return total_profit, total_spoilage_cost, service_level, df_profit, df_inv


    # print the financial statement
    print("\n" + "=" * 60)
    print(" Inventory Optimization Strategy Backtesting Report")
    print("=" * 60)

    # strategy A
    profit_q50, spoil_q50, sl_q50, df_res_q50, df_inv_q50 = simulate_strategy(df, 'pred_q50', 'q50')
    df = df.join(df_res_q50).join(df_inv_q50)

    # strategy B
    profit_q90, spoil_q90, sl_q90, df_res_q90, df_inv_q90 = simulate_strategy(df, 'pred_q90', 'q90')
    df = df.join(df_res_q90).join(df_inv_q90)
    
    if profit_q50 != 0:
        profit_uplift = ((profit_q90 - profit_q50) / abs(profit_q50)) * 100
    else:
        profit_uplift = 0.0

    print(f" Strategy A (Median Inventory) Total Profit: ${profit_q50:,.2f}, Spoilage Cost: ${spoil_q50:,.2f}")
    print(f" Strategy B (P90 Intelligent Inventory Management) Total Profit: ${profit_q90:,.2f}, Spoilage Cost: ${spoil_q90:,.2f}")
    print(f" Profit increase percentage: {profit_uplift:.2f}%")
    print("-" * 50)
    print(f" Strategy A has no shortage rate (service level): {sl_q50:.2f}%")
    print(f" Strategy B has no shortage rate (service level): {sl_q90:.2f}%")

    df.to_parquet(f'{processed_data_path}/m5_final_backtest.parquet', index=False)


if __name__ == "__main__":
    inventory_backtest()
