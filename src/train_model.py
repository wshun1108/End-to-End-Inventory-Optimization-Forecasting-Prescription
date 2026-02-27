import pandas as pd
import lightgbm as lgb
import joblib

processed_data_path = '../data/processed'
model_save_path = '../data/'

def train_model():
    df = pd.read_parquet(f'{processed_data_path}/m5_features.parquet')

    # prepare features columns
    drop_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'sales', 'date', 'wm_yr_wk']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # confirm that all feature columns are categorical
    cat_feats = ['weekday', 'event_name_1', 'event_type_1']
    for c in cat_feats:
        if c in df.columns:
            df[c] = df[c].astype('category')
    print(f'selected features ({len(feature_cols)}): {feature_cols}')

    # split data into train and validation sets
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=28)

    train_set = df[df['date'] <= split_date]
    test_set = df[df['date'] > split_date]

    X_train = train_set[feature_cols]
    y_train = train_set['sales']
    X_test = test_set[feature_cols]
    y_test = test_set['sales']

    # create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)

    # train quantile model
    quantile = [0.5, 0.9]
    models = {}

    for q in quantile:
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1

        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_test],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        models[q] = model
        joblib.dump(model, f'{model_save_path}lgb_model_q{int(q * 100)}.pkl')

    # generate predictions for test set
    # put the results in test_set in order to backtest
    test_set = test_set.copy()

    for q, model in models.items():
        test_set[f'pred_q{int(q * 100)}'] = model.predict(X_test)

    output_cols = ['id', 'item_id', 'store_id', 'date', 'sales', 'sell_price'] + [f'pred_q{int(q * 100)}' for q in quantile]
    final_output = test_set[output_cols]

    save_path = f'{processed_data_path}/m5_predictions.parquet'
    final_output.to_parquet(save_path, index=False)
    print(f"Finish prediction, save in: {save_path}")

if __name__ == "__main__":
    train_model()