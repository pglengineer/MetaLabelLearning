import json
import os

file_path = 'c:/Users/a124a/OneDrive/桌面/策略開發/MetaLabelLearning/MetaGranville/回測.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New BacktestEngine Code (Partial - just the class definition and backtest method, preserving other methods if possible, but replacing the whole class is easier if I have the full code. 
# Since I only want to modify backtest method, I can try to replace the cell content if the cell only contains BacktestEngine.
# Looking at the file, BacktestEngine is in a cell (lines 107-802 in original view, but it's huge).
# I will replace the entire cell content with the updated BacktestEngine class.
# I need to be careful to include all methods.
# I will read the original cell content, and replace the `backtest` method within it using string replacement.

# Find BacktestEngine cell
cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class BacktestEngine:" in source:
            cell_idx = i
            break

if cell_idx != -1:
    source = "".join(nb['cells'][cell_idx]['source'])
    
    # Replace backtest method signature
    old_sig = "def backtest(self, df: pd.DataFrame, signals: pd.Series, \n                 leverage_series: pd.Series = None) -> pd.DataFrame:"
    new_sig = "def backtest(self, df: pd.DataFrame, signals: pd.Series, \n                 leverage_series: pd.Series = None, side_series: pd.Series = None) -> pd.DataFrame:"
    
    if old_sig in source:
        source = source.replace(old_sig, new_sig)
    else:
        # Try looser match
        old_sig_loose = "def backtest(self, df: pd.DataFrame, signals: pd.Series,"
        if old_sig_loose in source:
             # This is harder to replace safely with string replace if formatting differs.
             # I'll replace the whole class logic for `backtest` method.
             pass

    # Actually, replacing the whole method body is safer.
    # I will construct the new cell source.
    # I'll use the full code I prepared.
    
    new_backtest_method = """    def backtest(self, df: pd.DataFrame, signals: pd.Series, 
                 leverage_series: pd.Series = None, side_series: pd.Series = None) -> pd.DataFrame:
        \"\"\"
        執行回測（支援動態槓桿調整 + 做空）
        
        Parameters:
        -----------
        df : pd.DataFrame
            價格資料（需包含 'Close', 'High', 'Low' 欄位）
        signals : pd.Series
            進場信號（True 表示進場點，需與 df 的 index 對齊）
        leverage_series : pd.Series, optional
            每筆交易的槓桿倍數（需與 df 的 index 對齊）
            如果為None，則所有交易使用base_leverage
        side_series : pd.Series, optional
            每筆交易的方向（1=做多, -1=做空），需與 df 的 index 對齊
            如果為None，則預設為 1 (做多)
            
        Returns:
        --------
        results : pd.DataFrame
            回測結果，包含每筆交易的詳細資訊
        \"\"\"
        df = df.copy()
        signals = signals.copy()
        
        # 保存價格數據供後續使用（用於 buy and hold 計算）
        self.price_df = df.copy()
        
        # 確保 signals 與 df 的 index 對齊
        if not signals.index.equals(df.index):
            signals = signals.reindex(df.index, fill_value=False)
        
        # 處理槓桿倍數
        if leverage_series is None:
            leverage_series = pd.Series(self.base_leverage, index=df.index)
        else:
            leverage_series = leverage_series.copy()
            if not leverage_series.index.equals(df.index):
                leverage_series = leverage_series.reindex(df.index, fill_value=self.base_leverage)
                
        # 處理方向
        if side_series is None:
            side_series = pd.Series(1, index=df.index)
        else:
            side_series = side_series.copy()
            if not side_series.index.equals(df.index):
                side_series = side_series.reindex(df.index, fill_value=1)
        
        # 初始化狀態
        position = None  # None: 無持倉, dict: {entry_time, entry_price, side, units, leverage, capital_used}
        trades = []
        current_equity = self.initial_capital  # 當前總equity
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_close = row['Close']
            current_high = row['High']
            current_low = row['Low']
            has_signal = signals.loc[timestamp]
            leverage = leverage_series.loc[timestamp]
            
            # ============================================
            # 邏輯1: 檢查進場條件（確保同時只會進場一次）
            # ============================================
            if has_signal and position is None:
                # 取得方向
                entry_side = side_series.loc[timestamp]
                if pd.isna(entry_side): entry_side = 1
                
                # 計算可進場的單位數
                # 可用資金 = 當前總equity * 槓桿倍數
                available_capital = current_equity * leverage
                
                # 計算可買入的單位數（假設1單位 = 1價格）
                # 實際應用中可能需要根據合約規格調整
                units = (available_capital / current_close) * 1
                
                # 實際使用的資金（不考慮槓桿時的資金）
                capital_used = current_equity
                
                # 以收盤價進場
                position = {
                    'entry_time': timestamp,
                    'entry_price': current_close,
                    'entry_idx': i,
                    'side': entry_side,
                    'units': units,  # 進場單位數
                    'leverage': leverage,  # 使用的槓桿倍數
                    'capital_used': capital_used  # 使用的資金
                }
                continue
            
            # ============================================
            # 邏輯2: 如果有持倉，檢查停損停利
            # ============================================
            if position is not None:
                entry_price = position['entry_price']
                entry_idx = position['entry_idx']
                side = position['side']
                units = position['units']
                leverage_used = position['leverage']
                capital_used = position['capital_used']
                
                # 計算停損價格
                stop_loss_price = entry_price * (1 - self.stop_loss_pct * side)
                
                # 計算停利價格
                take_profit_price = entry_price * (1 + self.take_profit_pct * side)
                
                # 計算持倉K線數
                bars_held = i - entry_idx
                
                # 檢查停損
                hit_stop_loss = False
                exit_price = None
                exit_reason = None
                
                if side == 1:  # 做多
                    if current_low <= stop_loss_price:
                        hit_stop_loss = True
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                elif side == -1: # 做空
                    if current_high >= stop_loss_price:
                        hit_stop_loss = True
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                
                # 檢查百分比停利
                hit_take_profit_pct = False
                if not hit_stop_loss:
                    if side == 1:  # 做多
                        if current_high >= take_profit_price:
                            hit_take_profit_pct = True
                            exit_price = take_profit_price
                            exit_reason = 'Take Profit (0.25%)'
                    elif side == -1: # 做空
                        if current_low <= take_profit_price:
                            hit_take_profit_pct = True
                            exit_price = take_profit_price
                            exit_reason = 'Take Profit (0.25%)'
                
                # 檢查時間停利（32根K後以收盤價平倉）
                hit_take_profit_bars = False
                if bars_held >= self.take_profit_bars:
                    hit_take_profit_bars = True
                    # 如果還沒觸發百分比停利，才使用時間停利
                    if not hit_take_profit_pct and not hit_stop_loss:
                        exit_price = current_close
                        exit_reason = 'Take Profit (32 bars)'
                
                # 如果觸發停損或停利，執行平倉
                if hit_stop_loss or hit_take_profit_pct or hit_take_profit_bars:
                    # 計算實際盈虧金額
                    # 盈虧 = (出場價 - 進場價) * 單位數 * 方向
                    pnl = (exit_price - entry_price - entry_price*0.0001) * units * side 
                    
                    # 計算報酬率（相對於使用的資金）
                    return_pct = (pnl / capital_used) * 100 if capital_used > 0 else 0
                    
                    # 更新當前總equity
                    current_equity = current_equity + pnl
                    
                    # 記錄交易
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': side,
                        'units': units,
                        'leverage': leverage_used,
                        'capital_used': capital_used,
                        'pnl': pnl,  # 盈虧金額
                        'return_pct': return_pct,  # 報酬率
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'equity_after': current_equity  # 平倉後總equity
                    }
                    trades.append(trade)
                    
                    # 重置持倉
                    position = None
        
        # 處理最後一筆未平倉的交易（如果有的話）
        if position is not None:
            last_timestamp = df.index[-1]
            last_close = df['Close'].iloc[-1]
            entry_price = position['entry_price']
            side = position['side']
            units = position['units']
            leverage_used = position['leverage']
            capital_used = position['capital_used']
            bars_held = len(df) - 1 - position['entry_idx']
            
            pnl = (last_close - entry_price) * units * side
            return_pct = (pnl / capital_used) * 100 if capital_used > 0 else 0
            current_equity = current_equity + pnl
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': last_timestamp,
                'entry_price': entry_price,
                'exit_price': last_close,
                'side': side,
                'units': units,
                'leverage': leverage_used,
                'capital_used': capital_used,
                'pnl': pnl,
                'return_pct': return_pct,
                'bars_held': bars_held,
                'exit_reason': 'End of Data',
                'stop_loss_price': entry_price * (1 - self.stop_loss_pct * side),
                'take_profit_price': entry_price * (1 + self.take_profit_pct * side),
                'equity_after': current_equity
            }
            trades.append(trade)
        
        # 轉換為 DataFrame
        if trades:
            results = pd.DataFrame(trades)
            self.trades = results
            self.final_equity = current_equity
            return results
        else:
            print("⚠️ 沒有產生任何交易")
            self.final_equity = current_equity
            return pd.DataFrame()
"""
    
    # Split source into lines to find start and end of backtest method
    lines = source.splitlines()
    start_line = -1
    end_line = -1
    
    for i, line in enumerate(lines):
        if "def backtest(self, df: pd.DataFrame, signals: pd.Series," in line:
            start_line = i
            break
    
    if start_line != -1:
        # Find end of method (next def or class end)
        for i in range(start_line + 1, len(lines)):
            if lines[i].strip().startswith("def "):
                end_line = i
                break
        if end_line == -1:
            end_line = len(lines)
            
        # Replace lines
        new_lines = lines[:start_line] + new_backtest_method.splitlines() + lines[end_line:]
        
        # Reconstruct source
        # Note: notebook source is a list of strings, each ending with \n usually
        new_source_list = [line + "\n" for line in new_lines]
        # Fix last line newline if needed
        if new_source_list:
            new_source_list[-1] = new_source_list[-1].rstrip("\n")
            
        nb['cells'][cell_idx]['source'] = new_source_list
        print("Updated BacktestEngine.backtest method.")
    else:
        print("Could not find backtest method start.")

# Update Usage Example to pass side_series
usage_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "strategy = S1Strategy(entry_param=0.5, entry_param2=0.4, rolling_window=460)" in source:
            usage_cell_idx = i
            break

if usage_cell_idx != -1:
    new_source = []
    for line in nb['cells'][usage_cell_idx]['source']:
        if "signals=df_with_signals['signal']" in line:
            new_source.append(line)
            new_source.append("    side_series=df_with_signals['side']  # Pass side\n")
        else:
            new_source.append(line)
    nb['cells'][usage_cell_idx]['source'] = new_source
    print("Updated Usage Example to pass side_series.")
else:
    print("Could not find Usage Example cell.")

# Save notebook
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
