import pandas as pd
import numpy as np

def inventory_backtest():

    processed_data_path = '../data/processed'
    df = pd.read_parquet(f'{processed_data_path}/m5_predictions.parquet')
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

    # function of calculate profit
    def simulate_strategy(data, policy_col):
        total_profit = 0
        stockout_days = 0
        total_spoilage_cost = 0
        total_days = len(data)

        for item_id, group in data.groupby('item_id'):
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

        service_level = (1 - (stockout_days / total_days)) * 100

        return total_profit, total_spoilage_cost, service_level


    # print the financial statement
    print("\n" + "=" * 60)
    print(" Inventory Optimization Strategy Backtesting Report")
    print("=" * 60)

    profit_q50, spoil_q50, sl_q50 = simulate_strategy(df, 'pred_q50')
    profit_q90, spoil_q90, sl_q90 = simulate_strategy(df, 'pred_q90')
    profit_uplift = ((profit_q90 - profit_q50) / abs(profit_q50)) * 100

    print(f" Strategy A (Median Inventory) Total Profit: ${profit_q50:,.2f}, Spoilage Cost: ${spoil_q50:,.2f}")
    print(f" Strategy B (P90 Intelligent Inventory Management) Total Profit: ${profit_q90:,.2f}, Spoilage Cost: ${spoil_q90:,.2f}")
    print(f" Profit increase percentage: {profit_uplift:.2f}%")
    print("-" * 50)
    print(f" Strategy A has no shortage rate (service level): {sl_q50:.2f}%")
    print(f" Strategy B has no shortage rate (service level): {sl_q90:.2f}%")

    df.to_parquet(f'{processed_data_path}/m5_final_backtest.parquet', index=False)


if __name__ == "__main__":
    inventory_backtest()




