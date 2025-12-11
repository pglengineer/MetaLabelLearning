# =====================================================
# 整合 RollingPredictor 模組
# 請將此代碼複製到您的 Jupyter Notebook 中執行
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

class RollingPredictor:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=-1, random_state=42),
            'catboost': CatBoostClassifier(n_estimators=100, verbose=0, thread_count=-1, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'et': ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'histgbm': HistGradientBoostingClassifier(random_state=42),
            'lr': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(max_iter=500, random_state=42),
            'ada': AdaBoostClassifier(n_estimators=100, random_state=42),
            'nb': GaussianNB(),
            'dt': DecisionTreeClassifier(random_state=42)
        }

    def run_rolling(self, X, y, model_name, first_test_year, roll_years, save_path):
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return pd.DataFrame()
        
        model = self.models[model_name]
        all_preds = []
        
        print(f"Starting rolling prediction for {model_name}...")
        
        # Ensure X index is timezone-aware if needed
        tz_info = X.index.tz
        
        for i in range(roll_years):
            year = first_test_year + i
            test_start_dt = pd.Timestamp(f'{year}-01-01').tz_localize(tz_info) if tz_info else pd.Timestamp(f'{year}-01-01')
            test_end_dt = test_start_dt + pd.DateOffset(years=1)
            
            # Train window: from 2014-11-01 (as requested) to test_start_dt
            train_start_dt = pd.Timestamp('2014-11-01').tz_localize(tz_info) if tz_info else pd.Timestamp('2014-11-01')
            
            train_mask = (X.index < test_start_dt) & (X.index >= train_start_dt)
            test_mask = (X.index >= test_start_dt) & (X.index < test_end_dt)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                print(f"Skipping {year} due to empty data (Train: {train_mask.sum()}, Test: {test_mask.sum()}).")
                continue
                
            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]
            
            print(f"Training for year {year} (Train: {len(X_train)}, Test: {len(X_test)})...")
            try:
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)
                else:
                    probs = None
                    
                preds = model.predict(X_test)
                
                # Create result DF
                res = pd.DataFrame(index=X_test.index)
                res['predicted_label'] = preds
                res['true_label'] = y_test
                if probs is not None:
                    for c in range(probs.shape[1]):
                        res[f'prob_class_{c}'] = probs[:, c]
                
                all_preds.append(res)
            except Exception as e:
                print(f"Error training {model_name} for year {year}: {e}")
            
        if not all_preds:
            return pd.DataFrame()
            
        final_df = pd.concat(all_preds)
        # Create directory if not exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(save_path)
        print(f"Saved predictions to {save_path}")
        return final_df

# 執行滾動預測
# 請確保 X 和 y_mapped 已經在 Notebook 中定義
if 'X' in locals() and 'y_mapped' in locals():
    predictor = RollingPredictor()

    # 定義要執行的模型
    models_to_run = ['xgboost', 'lightgbm', 'catboost', 'rf', 'et', 'gb', 'histgbm', 'lr', 'svm', 'mlp', 'ada', 'nb', 'dt']

    # 設定參數
    first_test_year = 2020
    roll_years = 6 # 2020, 2021, 2022, 2023, 2024, 2025
    results_dir = Path('pred_results_rolling')

    print(f"開始執行滾動預測，測試年份: {first_test_year} 到 {first_test_year + roll_years - 1}")

    for m in models_to_run:
        save_p = results_dir / f'pred_{m}_{first_test_year}_{first_test_year + roll_years - 1}.csv'
        # 使用 y_mapped (編碼後的標籤) 進行訓練
        try:
            predictor.run_rolling(X, y_mapped, m, first_test_year, roll_years, str(save_p))
        except Exception as e:
            print(f"Failed to run for {m}: {e}")
else:
    print("錯誤: X 或 y_mapped 未定義。請先執行 Notebook 前面的單元格以載入數據。")
