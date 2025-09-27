from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from functools import lru_cache
import os
import requests
import io
import json
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mageball_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# LotteryAnalyzer 类（适配 Mega Millions）
WHITE_MAX = 70
PB_MAX = 25

def get_latest_draw_info():
    """获取最新开奖信息和下期开奖时间"""
    try:
        # 从在线数据源获取最新数据
        response = requests.get('https://data.ny.gov/api/views/5xaw-6ayf/rows.csv', timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # 解析最新一期数据
        if len(df) > 0:
            latest_row = df.iloc[0]  # 第一行是最新的
            draw_date = latest_row['Draw Date']
            winning_numbers = latest_row['Winning Numbers']
            mega_ball = latest_row['Mega Ball']
            
            # 解析开奖日期
            draw_date_parsed = pd.to_datetime(draw_date, format='%m/%d/%Y', errors='coerce')
            
            # 计算下期开奖时间（Mega Millions 每周二和周五晚上11点开奖 EST）
            next_draw_date = get_next_draw_date(draw_date_parsed)
            
            return {
                'draw_date': draw_date,
                'winning_numbers': winning_numbers,
                'mega_ball': mega_ball,
                'next_draw_date': next_draw_date,
                'total_records': len(df)
            }
    except Exception as e:
        print(f"获取最新开奖信息失败: {e}")
    
    return None

def get_next_draw_date(last_draw_date):
    """计算下期开奖时间"""
    if pd.isna(last_draw_date):
        # 如果解析失败，使用当前时间计算
        now = datetime.now()
    else:
        now = datetime.now()
        last_draw = last_draw_date.to_pydatetime()
    
    # Mega Millions 开奖时间：每周二(1)和周五(4) 晚上11点 EST
    # 这里使用简化计算，实际时区可能需要调整
    current_weekday = now.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
    
    # 计算到下次开奖的天数
    if current_weekday < 1:  # Monday
        days_until_next = 1 - current_weekday
    elif current_weekday == 1:  # Tuesday
        # 如果是周二，检查是否已过开奖时间
        if now.hour >= 23:
            days_until_next = 3  # 下周五
        else:
            days_until_next = 0  # 今天
    elif current_weekday < 4:  # Wednesday or Thursday
        days_until_next = 4 - current_weekday
    elif current_weekday == 4:  # Friday
        # 如果是周五，检查是否已过开奖时间
        if now.hour >= 23:
            days_until_next = 4  # 下周二
        else:
            days_until_next = 0  # 今天
    else:  # Weekend
        days_until_next = 1 + (7 - current_weekday)  # 下周二
    
    next_draw = now + timedelta(days=days_until_next)
    next_draw = next_draw.replace(hour=23, minute=0, second=0, microsecond=0)
    
    return next_draw

class LotteryAnalyzer:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.white_cols: List[str] = []
        self.pb_col: Optional[str] = None
        self.features_white: Optional[pd.DataFrame] = None
        self.features_pb: Optional[pd.DataFrame] = None
        self.scores_combined: Optional[pd.DataFrame] = None
        self.weights = (0.3, 0.3, 0.2, 0.2)  # 频率、近期、间隔、回测性能
        self.prediction_history_file = 'uploads/prediction_history.json'
        os.makedirs('uploads', exist_ok=True)

    def load_csv(self, file_path: str = None, url: str = None) -> "LotteryAnalyzer":
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                print(f"从 {url} 加载数据：{len(df)} 行")
            except Exception as e:
                raise ValueError(f"无法从 {url} 下载数据：{str(e)}")
        else:
            df = pd.read_csv(file_path)
            print(f"从本地加载数据：{len(df)} 行")

        print(f"原始列名：{df.columns.tolist()}")
        print(f"前5行 Draw Date：{df['Draw Date'].head().tolist()}")
        print(f"前5行 Mega Ball：{df['Mega Ball'].head().tolist()}")
        cols_lower = [c.lower().strip() for c in df.columns]

        # 查找日期列
        date_col_candidates = [c for c in df.columns if 'date' in c.lower()]
        date_col = date_col_candidates[0] if date_col_candidates else None

        # 查找 Winning Numbers 和 Mega Ball 列
        wn_col_candidates = [c for c in df.columns if 'winning numbers' in c.lower()]
        wn_col = wn_col_candidates[0] if wn_col_candidates else None
        pb_col_candidates = [c for c in df.columns if 'mega ball' in c.lower()]
        pb_col = pb_col_candidates[0] if pb_col_candidates else None

        if wn_col and pb_col:
            # 解析日期
            if date_col:
                df['date'] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
                invalid_dates = df['date'].isna().sum()
                print(f"无效日期行数：{invalid_dates}")
                if invalid_dates > 0:
                    print(f"无效日期示例：{df[date_col][df['date'].isna()].head().tolist()}")
            else:
                df['date'] = pd.NaT
                print("警告：未找到日期列，禁用日期过滤")

            # 解析 Winning Numbers 为白球
            def parse_row(s):
                parts = str(s).strip().split()
                if len(parts) >= 5:
                    return parts[:5]
                return [np.nan] * 5

            parsed_white = df[wn_col].astype(str).apply(parse_row).apply(pd.Series)
            parsed_white.columns = ['n1', 'n2', 'n3', 'n4', 'n5']
            df = pd.concat([df, parsed_white], axis=1)
            self.white_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
            self.pb_col = 'megaball'

            # 使用 CSV 中的 Mega Ball 列
            df['megaball'] = pd.to_numeric(df[pb_col], errors='coerce')
            print(f"Mega Ball 非空行数：{len(df[df['megaball'].notna()])}")
        else:
            raise ValueError("无效的 CSV 格式。请确保包含 'Winning Numbers' 和 'Mega Ball' 列。")

        # 处理 NaN
        initial_rows = len(df)
        df = df.dropna(subset=['date'])  # 仅移除无效日期行
        print(f"移除无效日期后：{len(df)} 行（移除了 {initial_rows - len(df)} 行）")
        for c in self.white_cols + [self.pb_col]:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('Int64')

        # 日期过滤 - 2017年10月31日是Mega Millions规则变更日期
        if not df['date'].isna().all():
            initial_rows = len(df)
            df = df[df['date'] >= '2017-10-31']
            print(f"过滤 2017-10-31 后数据：{len(df)} 行（移除了 {initial_rows - len(df)} 行）")
        else:
            print("警告：所有日期无效，跳过 2017-10-31 过滤")

        # 验证号码范围
        for c in self.white_cols:
            if not df[c].isna().all() and not ((df[c] >= 1) & (df[c] <= WHITE_MAX)).all():
                raise ValueError(f"白球列 {c} 包含超出 1-{WHITE_MAX} 的值")
        if not df[self.pb_col].isna().all() and not ((df[self.pb_col] >= 1) & (df[self.pb_col] <= PB_MAX)).all():
            raise ValueError(f"红球列包含超出 1-{PB_MAX} 的值")

        df = df.sort_values('date', na_position='last').reset_index(drop=True)
        print(f"最终有效数据：{len(df)} 行")
        self.df = df
        return self

    def backtest_strategy(self, strategy: str = 'hybrid', n_tickets: int = 10, window: int = 200, constraints: Optional[Dict] = None, random_state: Optional[int] = None, step: int = 20) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
        if self.df is None:
            raise ValueError("请先使用 load_csv() 加载数据。")
        window = min(window, max(50, len(self.df) // 10))
        print(f"调整回测窗口大小为：{window}")
        if len(self.df) < window + 1:
            raise ValueError(f"数据不足以进行回测 ({len(self.df)} < {window + 1})。请加载更多数据。")
        rng = np.random.default_rng(random_state)
        records = []
        white_perf = pd.Series(0, index=range(1, WHITE_MAX + 1), dtype=int)
        pb_perf = pd.Series(0, index=range(1, PB_MAX + 1), dtype=int)
        
        # 新增：红球命中概率统计
        total_predictions = 0
        total_pb_hits = 0
        pb_hit_stats = {}
        
        for t in range(window, len(self.df), step):
            train = self.df.iloc[:t].copy()
            test_row = self.df.iloc[t]
            la_temp = LotteryAnalyzer()
            la_temp.df = train
            la_temp.white_cols = self.white_cols
            la_temp.pb_col = self.pb_col
            la_temp.compute_features(recent_window=50, backtest_window=window, use_backtest=False)
            picks = la_temp.generate_picks(n_tickets=n_tickets, strategy=strategy, constraints=constraints, random_state=int(rng.integers(0, 1000000000)))
            truth_whites = set([test_row[c] for c in self.white_cols if pd.notna(test_row[c])])
            truth_pb = test_row[self.pb_col]
            if pd.isna(truth_pb) or truth_pb == 0:
                continue
            
            for _, row in picks.iterrows():
                whites = set([row['n1'], row['n2'], row['n3'], row['n4'], row['n5']])
                pb = row['megaball']
                white_matches = len(whites & truth_whites)
                pb_match = int(pb == truth_pb)
                
                # 记录所有的结果（不仅仅是红球命中的）
                records.append({"t": t, "white_matches": white_matches, "pb_match": pb_match})
                
                # 统计红球命中情况
                total_predictions += 1
                if pb_match:
                    total_pb_hits += 1
                    pb_perf[pb] += 1
                
                # 统计白球表现
                for w in whites:
                    if w in truth_whites:
                        white_perf[w] += 1
        
        # 计算红球整体命中概率
        pb_hit_probability = total_pb_hits / total_predictions if total_predictions > 0 else 0
        pb_hit_stats = {
            'total_predictions': total_predictions,
            'total_pb_hits': total_pb_hits,
            'pb_hit_probability': pb_hit_probability,
            'theoretical_probability': 1/PB_MAX,  # 理论概率 1/25
            'performance_ratio': pb_hit_probability / (1/PB_MAX) if (1/PB_MAX) > 0 else 0
        }
        
        out = pd.DataFrame(records)
        summary = out.groupby(["white_matches", "pb_match"]).size().rename("count").reset_index()
        if summary.empty:
            summary = pd.DataFrame({"white_matches": [0], "pb_match": [0], "count": [0]})
        return summary, white_perf, pb_perf, pb_hit_stats

    @lru_cache(maxsize=128)
    def _compute_features_cached(self, recent_window: int, data_hash: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        total = len(self.df)
        white_counts = pd.Series(0, index=range(1, WHITE_MAX + 1), dtype=int)
        for c in self.white_cols:
            vc = self.df[c].value_counts().reindex(range(1, WHITE_MAX + 1)).fillna(0).astype(int)
            white_counts = white_counts.add(vc, fill_value=0).astype(int)
        pb_counts = self.df[self.pb_col].value_counts().reindex(range(1, PB_MAX + 1)).fillna(0).astype(int)

        window = min(recent_window, total)
        recent = self.df.tail(window)
        rec_w = pd.Series(0, index=range(1, WHITE_MAX + 1), dtype=int)
        for c in self.white_cols:
            vc = recent[c].value_counts().reindex(range(1, WHITE_MAX + 1)).fillna(0).astype(int)
            rec_w = rec_w.add(vc, fill_value=0).astype(int)
        rec_pb = recent[self.pb_col].value_counts().reindex(range(1, PB_MAX + 1)).fillna(0).astype(int)

        gap_w = pd.Series(np.inf, index=range(1, WHITE_MAX + 1))
        gap_pb = pd.Series(np.inf, index=range(1, PB_MAX + 1))
        last_seen_w = pd.Series(np.nan, index=range(1, WHITE_MAX + 1))
        last_seen_pb = pd.Series(np.nan, index=range(1, PB_MAX + 1))
        for num in range(1, WHITE_MAX + 1):
            mask = self.df[self.white_cols].eq(num).any(axis=1)
            if mask.any():
                last_seen_w[num] = self.df.index[mask][::-1][0]
        for num in range(1, PB_MAX + 1):
            mask = self.df[self.pb_col].eq(num)
            if mask.any():
                last_seen_pb[num] = self.df.index[mask][::-1][0]
        gap_w = (len(self.df) - last_seen_w).fillna(len(self.df)).astype(int)
        gap_pb = (len(self.df) - last_seen_pb).fillna(len(self.df)).astype(int)

        fw = pd.DataFrame({'freq': white_counts, 'recent': rec_w, 'gap': gap_w, 'performance': pd.Series(0, index=range(1, WHITE_MAX + 1), dtype=int)})
        fpb = pd.DataFrame({'freq': pb_counts, 'recent': rec_pb, 'gap': gap_pb, 'performance': pd.Series(0, index=range(1, PB_MAX + 1), dtype=int)})
        return fw, fpb

    def compute_features(self, recent_window: int = 50, backtest_window: int = 200, use_backtest: bool = True) -> "LotteryAnalyzer":
        if self.df is None:
            raise ValueError("请先使用 load_csv() 加载数据。")
        backtest_window = min(backtest_window, max(50, len(self.df) // 10))
        print(f"调整特征计算窗口大小为：{backtest_window}")
        data_hash = hash(self.df.to_string())
        fw, fpb = self._compute_features_cached(recent_window, data_hash)

        if use_backtest:
            _, white_perf, pb_perf, _ = self.backtest_strategy(strategy='hybrid', n_tickets=10, window=backtest_window, step=20)
            fw['performance'] = white_perf
            fpb['performance'] = pb_perf
        else:
            fw['performance'] = pd.Series(0, index=range(1, WHITE_MAX + 1), dtype=int)
            fpb['performance'] = pd.Series(0, index=range(1, PB_MAX + 1), dtype=int)

        def normalize_df(df):
            df = df.astype(float)
            result = df.copy()
            for col in df.columns:
                s = df[col]
                if s.max() == s.min():
                    result[col] = 0.0
                else:
                    result[col] = (s - s.min()) / (s.max() - s.min())
            return result

        fw_norm = normalize_df(fw)
        fpb_norm = normalize_df(fpb)

        w_freq, w_recent, w_gap, w_perf = self.weights
        white_score = fw_norm['freq'] * w_freq + fw_norm['recent'] * w_recent + fw_norm['gap'] * w_gap + fw_norm['performance'] * w_perf
        pb_score = fpb_norm['freq'] * w_freq + fpb_norm['recent'] * w_recent + fpb_norm['gap'] * w_gap + fpb_norm['performance'] * w_perf

        self.features_white = fw.assign(freq_norm=fw_norm['freq'], recent_norm=fw_norm['recent'],
                                       gap_norm=fw_norm['gap'], performance_norm=fw_norm['performance'], score=white_score)
        self.features_pb = fpb.assign(freq_norm=fpb_norm['freq'], recent_norm=fpb_norm['recent'],
                                      gap_norm=fpb_norm['gap'], performance_norm=fpb_norm['performance'], score=pb_score)

        whites_df = self.features_white.reset_index().rename(columns={'index': 'number'}).assign(type='white')
        pbs_df = self.features_pb.reset_index().rename(columns={'index': 'number'}).assign(type='megaball')
        self.scores_combined = pd.concat([whites_df[['number', 'score', 'type']], pbs_df[['number', 'score', 'type']]], ignore_index=True)
        return self

    def rank_numbers(self, topn: Optional[int] = None) -> pd.DataFrame:
        if self.scores_combined is None:
            raise ValueError("请先调用 compute_features()。")
        df = self.scores_combined.sort_values('score', ascending=False).reset_index(drop=True)
        if topn:
            return df.head(topn)
        return df

    def generate_picks(self, n_tickets: int = 5, strategy: str = 'hybrid',
                       constraints: Optional[Dict] = None, random_state: Optional[int] = None) -> pd.DataFrame:
        if self.scores_combined is None:
            self.compute_features()
        rng = np.random.default_rng(random_state)

        w_scores = self.features_white['score'].reindex(range(1, WHITE_MAX + 1)).fillna(0).astype(float).values
        p_scores = self.features_pb['score'].reindex(range(1, PB_MAX + 1)).fillna(0).astype(float).values

        if strategy == 'hot':
            w_probs = np.where(w_scores <= 0, 1e-6, w_scores)
            p_probs = np.where(p_scores <= 0, 1e-6, p_scores)
        elif strategy == 'overdue':
            w_probs = self.features_white['gap'].reindex(range(1, WHITE_MAX + 1)).fillna(0).values
            p_probs = self.features_pb['gap'].reindex(range(1, PB_MAX + 1)).fillna(0).values
            w_probs = np.where(w_probs <= 0, 1e-6, w_probs)
            p_probs = np.where(p_probs <= 0, 1e-6, p_probs)
        elif strategy == 'random':
            w_probs = np.ones(WHITE_MAX)
            p_probs = np.ones(PB_MAX)
        else:
            w_probs = np.where(w_scores <= 0, 1e-6, w_scores)
            p_probs = np.where(p_scores <= 0, 1e-6, p_scores)

        w_probs = w_probs / w_probs.sum()
        p_probs = p_probs / p_probs.sum()

        picks = []
        tries = 0
        max_tries = 5000
        while len(picks) < n_tickets and tries < max_tries:
            tries += 1
            whites = rng.choice(np.arange(1, WHITE_MAX + 1), size=5, replace=False, p=w_probs)
            whites = np.sort(whites).tolist()
            pb = int(rng.choice(np.arange(1, PB_MAX + 1), p=p_probs))
            if self._satisfy_constraints(whites, pb, constraints):
                picks.append(whites + [pb])

        df_picks = pd.DataFrame(picks, columns=['n1', 'n2', 'n3', 'n4', 'n5', 'megaball'])
        return df_picks

    def _satisfy_constraints(self, whites: List[int], pb: int, constraints: Optional[Dict] = None) -> bool:
        if not constraints:
            return True
        total = sum(whites)
        odd = sum(1 for w in whites if w % 2 == 1)
        spread = max(whites) - min(whites)
        if 'sum_range' in constraints:
            lo, hi = constraints['sum_range']
            if not (lo <= total <= hi):
                return False
        if 'odd_even' in constraints:
            lo, hi = constraints['odd_even']
            if not (lo <= odd <= hi):
                return False
        if 'min_spread' in constraints:
            if spread < constraints['min_spread']:
                return False
        return True

    def save_picks_csv(self, df_picks: pd.DataFrame, path: str = 'recommended_picks.csv'):
        df_picks.to_csv(path, index=False)
    
    def save_prediction_to_history(self, picks: pd.DataFrame, strategy: str = 'hybrid', 
                                  constraints: Optional[Dict] = None, 
                                  additional_info: Optional[Dict] = None) -> None:
        """
        保存预测结果到历史记录文件
        
        Args:
            picks: 预测的号码组合DataFrame
            strategy: 使用的策略
            constraints: 约束条件
            additional_info: 额外信息（如模型参数等）
        """
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'constraints': constraints or {},
            'additional_info': additional_info or {},
            'predictions': picks.to_dict('records'),
            'data_rows_used': len(self.df) if self.df is not None else 0,
            'weights': self.weights
        }
        
        # 读取现有历史记录
        history = self.load_prediction_history()
        history.append(prediction_record)
        
        # 保存更新后的历史记录
        try:
            with open(self.prediction_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"预测记录已保存到: {self.prediction_history_file}")
        except Exception as e:
            print(f"保存预测历史记录失败: {e}")
    
    def load_prediction_history(self) -> List[Dict]:
        """
        加载历史预测记录
        
        Returns:
            历史预测记录列表
        """
        try:
            if os.path.exists(self.prediction_history_file):
                with open(self.prediction_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载预测历史记录失败: {e}")
        return []
    
    def backtest_historical_predictions(self, actual_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        对历史预测进行回测分析
        
        Args:
            actual_results: 实际开奖结果DataFrame，如果为None则使用当前数据
            
        Returns:
            回测结果DataFrame
        """
        history = self.load_prediction_history()
        if not history:
            print("没有找到历史预测记录")
            return pd.DataFrame()
        
        # 使用当前加载的数据作为实际结果
        if actual_results is None:
            if self.df is None:
                print("没有加载实际开奖数据")
                return pd.DataFrame()
            actual_results = self.df
        
        backtest_results = []
        
        for i, record in enumerate(history):
            timestamp = record['timestamp']
            predictions = record['predictions']
            strategy = record['strategy']
            
            # 解析预测时间
            try:
                pred_time = datetime.fromisoformat(timestamp)
            except:
                print(f"无法解析预测时间: {timestamp}")
                continue
            
            # 查找预测后的实际开奖结果（比预测时间晚的开奖）
            actual_after_pred = actual_results[actual_results['date'] > pred_time]
            
            if len(actual_after_pred) == 0:
                continue  # 没有后续开奖数据
            
            # 分析每个预测号码组合的表现
            for j, pred in enumerate(predictions):
                pred_whites = {pred['n1'], pred['n2'], pred['n3'], pred['n4'], pred['n5']}
                pred_pb = pred['megaball']
                
                # 统计在后续开奖中的命中情况
                total_draws = len(actual_after_pred)
                white_matches = []
                pb_matches = 0
                
                for _, actual_row in actual_after_pred.iterrows():
                    actual_whites = {actual_row[c] for c in self.white_cols if pd.notna(actual_row[c])}
                    actual_pb = actual_row[self.pb_col]
                    
                    # 计算白球匹配数
                    white_match_count = len(pred_whites & actual_whites)
                    white_matches.append(white_match_count)
                    
                    # 检查红球匹配
                    if pred_pb == actual_pb:
                        pb_matches += 1
                
                # 计算统计指标
                max_white_matches = max(white_matches) if white_matches else 0
                avg_white_matches = np.mean(white_matches) if white_matches else 0
                pb_hit_rate = pb_matches / total_draws if total_draws > 0 else 0
                
                backtest_results.append({
                    'record_id': i,
                    'prediction_id': j,
                    'timestamp': timestamp,
                    'strategy': strategy,
                    'predicted_whites': sorted(list(pred_whites)),
                    'predicted_pb': pred_pb,
                    'total_subsequent_draws': total_draws,
                    'max_white_matches': max_white_matches,
                    'avg_white_matches': avg_white_matches,
                    'pb_hit_rate': pb_hit_rate,
                    'pb_hits': pb_matches
                })
        
        return pd.DataFrame(backtest_results)
    
    def get_prediction_performance_summary(self) -> Dict:
        """
        获取历史预测表现摘要
        
        Returns:
            包含各种统计指标的字典
        """
        backtest_results = self.backtest_historical_predictions()
        
        if backtest_results.empty:
            return {'error': '没有可用的回测数据'}
        
        # 按策略分组统计
        strategy_stats = backtest_results.groupby('strategy').agg({
            'max_white_matches': ['mean', 'max', 'std'],
            'avg_white_matches': 'mean',
            'pb_hit_rate': 'mean',
            'pb_hits': 'sum',
            'total_subsequent_draws': 'sum'
        }).round(4)
        
        # 整体统计
        overall_stats = {
            'total_predictions': len(backtest_results),
            'unique_strategies': backtest_results['strategy'].nunique(),
            'avg_max_white_matches': backtest_results['max_white_matches'].mean(),
            'best_white_matches': backtest_results['max_white_matches'].max(),
            'overall_pb_hit_rate': backtest_results['pb_hit_rate'].mean(),
            'total_pb_hits': backtest_results['pb_hits'].sum(),
            'total_draws_analyzed': backtest_results['total_subsequent_draws'].sum()
        }
        
        return {
            'overall_stats': overall_stats,
            'strategy_stats': strategy_stats.to_dict(),
            'detailed_results': backtest_results.to_dict('records')
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    result = None
    chart_data = None
    picks_file = None
    latest_draw_info = get_latest_draw_info()
    optimization_progress = None

    if request.method == 'POST':
        try:
            window = int(request.form.get('window', 200))
            n_tickets = int(request.form.get('n_tickets', 10))
            step = int(request.form.get('step', 20))
            
            # 是否启用参数优化
            enable_optimization = request.form.get('enable_optimization', 'off') == 'on'

            file = request.files.get('csv_file')
            la = LotteryAnalyzer()

            if file and file.filename:
                upload_dir = 'uploads'
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, 'mega_millions_clean.csv')
                file.save(file_path)
                la.load_csv(file_path=file_path)
            else:
                la.load_csv(url='https://data.ny.gov/api/views/5xaw-6ayf/rows.csv')

            start_time = datetime.now()
            
            # 如果启用参数优化，先进行参数优化分析
            if enable_optimization:
                print("开始参数优化分析...")
                
                # 参数优化范围（可以根据需要调整）
                window_range = range(100, 301, 50)  # 100, 150, 200, 250, 300
                step_range = range(10, 51, 10)      # 10, 20, 30, 40, 50
                
                optimization_results = []
                best_pb_hit_rate = 0
                best_params = None
                best_theoretical_ratio = 0
                best_theoretical_params = None
                
                total_combinations = len(list(window_range)) * len(list(step_range))
                current_combination = 0
                
                for opt_window in window_range:
                    for opt_step in step_range:
                        current_combination += 1
                        progress = (current_combination / total_combinations) * 100
                        
                        # 发送实时进度更新
                        socketio.emit('progress_update', {
                            'current': current_combination,
                            'total': total_combinations,
                            'progress': progress,
                            'status': f'正在测试参数组合: 窗口={opt_window}, 步长={opt_step}'
                        })
                        
                        # 添加短暂延迟以确保前端能接收到更新
                        socketio.sleep(0.1)
                        
                        print(f"参数优化进度: {progress:.1f}% ({current_combination}/{total_combinations})")
                        
                        try:
                            # 调整窗口大小
                            adjusted_window = min(opt_window, max(50, len(la.df) // 10))
                            
                            # 进行回测
                            summary, _, _, pb_hit_stats = la.backtest_strategy(
                                strategy='hybrid', 
                                n_tickets=n_tickets, 
                                window=adjusted_window, 
                                step=opt_step
                            )
                            
                            # 记录结果
                            result_entry = {
                                'window': opt_window,
                                'step': opt_step,
                                'adjusted_window': adjusted_window,
                                'pb_hit_probability': pb_hit_stats['pb_hit_probability'],
                                'performance_ratio': pb_hit_stats['performance_ratio']
                            }
                            
                            optimization_results.append(result_entry)
                            
                            # 更新最佳红球命中率参数
                            if pb_hit_stats['pb_hit_probability'] > best_pb_hit_rate:
                                best_pb_hit_rate = pb_hit_stats['pb_hit_probability']
                                best_params = result_entry.copy()
                            
                            # 更新最佳理论概率表现参数
                            if pb_hit_stats['performance_ratio'] > best_theoretical_ratio:
                                best_theoretical_ratio = pb_hit_stats['performance_ratio']
                                best_theoretical_params = result_entry.copy()
                                
                        except Exception as e:
                            print(f"参数组合 window={opt_window}, step={opt_step} 测试失败: {e}")
                            continue
                
                # 使用最佳参数进行最终分析
                if best_params:
                    window = best_params['window']
                    step = best_params['step']
                    print(f"使用最佳参数: window={window}, step={step}, 红球命中率={best_params['pb_hit_probability']:.4f}")
                    
                    optimization_progress = {
                        'total_combinations': total_combinations,
                        'best_params': best_params,
                        'best_theoretical_params': best_theoretical_params,
                        'optimization_results': optimization_results[:10]  # 只显示前10个结果
                    }
                else:
                    print("参数优化失败，使用默认参数")

            window = min(window, max(50, len(la.df) // 10))
            print(f"路由中调整窗口大小为：{window}")

            la.compute_features(recent_window=50, backtest_window=window, use_backtest=True)
            compute_time = (datetime.now() - start_time).total_seconds()

            summary, _, _, pb_hit_stats = la.backtest_strategy(strategy='hybrid', n_tickets=n_tickets, window=window, step=step)
            top_numbers = la.rank_numbers(topn=20)
            picks = la.generate_picks(
                n_tickets=5,
                strategy="hybrid",
                constraints={"sum_range": (100, 200), "odd_even": (2, 4), "min_spread": 10},
                random_state=2025
            )
            
            # 保存预测到历史记录
            la.save_prediction_to_history(
                picks=picks, 
                strategy="hybrid",
                constraints={"sum_range": (100, 200), "odd_even": (2, 4), "min_spread": 10},
                additional_info={
                    "window": window,
                    "n_tickets": n_tickets, 
                    "step": step,
                    "compute_time": f"{compute_time:.2f} 秒",
                    "optimization_enabled": enable_optimization,
                    "optimization_progress": optimization_progress
                }
            )

            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            picks_path = os.path.join(upload_dir, 'recommended_picks.csv')
            la.save_picks_csv(picks, picks_path)
            picks_file = picks_path

            summary['label'] = summary.apply(lambda x: f"{x['white_matches']}白+Mega Ball命中", axis=1)
            chart_labels = summary['label'].tolist()
            chart_values = summary['count'].tolist()

            chart_data = {
                'labels': chart_labels,
                'data': chart_values,
                'backgroundColor': [
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(255, 159, 64, 0.6)',
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(153, 102, 255, 0.6)',
                    'rgba(255, 205, 86, 0.6)'
                ],
                'borderColor': [
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 205, 86, 1)'
                ]
            }

            result = {
                'rows_loaded': len(la.df),
                'compute_time': f"{compute_time:.2f} 秒",
                'summary': summary.to_dict(orient='records'),
                'top_numbers': top_numbers.to_dict(orient='records'),
                'picks': picks.to_dict(orient='records'),
                'pb_hit_stats': pb_hit_stats,
                'optimization_progress': optimization_progress,
                'final_params': {'window': window, 'step': step} if enable_optimization else None
            }
            
            # 发送分析完成事件
            if enable_optimization:
                socketio.emit('analysis_complete', {'status': 'success'})

        except Exception as e:
            error = f"错误：{str(e)}"

    return render_template('index.html', error=error, result=result, chart_data=chart_data, picks_file=picks_file, latest_draw_info=latest_draw_info)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/prediction_history', methods=['GET'])
def prediction_history():
    """显示历史预测记录和回测结果"""
    try:
        la = LotteryAnalyzer()
        
        # 加载数据进行回测
        la.load_csv(url='https://data.ny.gov/api/views/5xaw-6ayf/rows.csv')
        
        # 获取历史预测记录
        history = la.load_prediction_history()
        
        # 进行回测分析
        backtest_results = la.backtest_historical_predictions()
        performance_summary = la.get_prediction_performance_summary()
        
        return render_template('prediction_history.html', 
                             history=history, 
                             backtest_results=backtest_results.to_dict('records') if not backtest_results.empty else [],
                             performance_summary=performance_summary)
                             
    except Exception as e:
        return render_template('prediction_history.html', 
                             error=f"错误：{str(e)}",
                             history=[], 
                             backtest_results=[],
                             performance_summary={})

@app.route('/parameter_optimization', methods=['GET', 'POST'])
def parameter_optimization():
    """参数优化分析 - 找到红球命中率最好的回测方案"""
    if request.method == 'GET':
        return render_template('parameter_optimization.html')
    
    try:
        # 获取参数范围
        window_start = int(request.form.get('window_start', 100))
        window_end = int(request.form.get('window_end', 300))
        window_step = int(request.form.get('window_step', 50))
        step_start = int(request.form.get('step_start', 10))
        step_end = int(request.form.get('step_end', 50))
        step_step = int(request.form.get('step_step', 10))
        n_tickets = int(request.form.get('n_tickets', 10))
        
        # 加载数据
        la = LotteryAnalyzer()
        file = request.files.get('csv_file')
        if file and file.filename:
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, 'mega_millions_clean.csv')
            file.save(file_path)
            la.load_csv(file_path=file_path)
        else:
            la.load_csv(url='https://data.ny.gov/api/views/5xaw-6ayf/rows.csv')
        
        # 参数组合列表
        window_range = range(window_start, window_end + 1, window_step)
        step_range = range(step_start, step_end + 1, step_step)
        
        optimization_results = []
        best_pb_hit_rate = 0
        best_params = None
        best_theoretical_ratio = 0
        best_theoretical_params = None
        
        total_combinations = len(list(window_range)) * len(list(step_range))
        current_combination = 0
        
        for window in window_range:
            for step in step_range:
                current_combination += 1
                print(f"测试参数组合 {current_combination}/{total_combinations}: window={window}, step={step}")
                
                try:
                    # 调整窗口大小
                    adjusted_window = min(window, max(50, len(la.df) // 10))
                    
                    # 进行回测
                    start_time = datetime.now()
                    summary, _, _, pb_hit_stats = la.backtest_strategy(
                        strategy='hybrid', 
                        n_tickets=n_tickets, 
                        window=adjusted_window, 
                        step=step
                    )
                    compute_time = (datetime.now() - start_time).total_seconds()
                    
                    # 记录结果
                    result = {
                        'window': window,
                        'step': step,
                        'adjusted_window': adjusted_window,
                        'pb_hit_probability': pb_hit_stats['pb_hit_probability'],
                        'total_predictions': pb_hit_stats['total_predictions'],
                        'total_pb_hits': pb_hit_stats['total_pb_hits'],
                        'performance_ratio': pb_hit_stats['performance_ratio'],
                        'compute_time': compute_time,
                        'summary_count': len(summary)
                    }
                    
                    optimization_results.append(result)
                    
                    # 更新最佳红球命中率参数
                    if pb_hit_stats['pb_hit_probability'] > best_pb_hit_rate:
                        best_pb_hit_rate = pb_hit_stats['pb_hit_probability']
                        best_params = {
                            'window': window,
                            'step': step,
                            'adjusted_window': adjusted_window,
                            'pb_hit_probability': pb_hit_stats['pb_hit_probability'],
                            'performance_ratio': pb_hit_stats['performance_ratio']
                        }
                    
                    # 更新最佳理论概率表现参数
                    if pb_hit_stats['performance_ratio'] > best_theoretical_ratio:
                        best_theoretical_ratio = pb_hit_stats['performance_ratio']
                        best_theoretical_params = {
                            'window': window,
                            'step': step,
                            'adjusted_window': adjusted_window,
                            'pb_hit_probability': pb_hit_stats['pb_hit_probability'],
                            'performance_ratio': pb_hit_stats['performance_ratio']
                        }
                    
                except Exception as e:
                    print(f"参数组合 window={window}, step={step} 测试失败: {e}")
                    optimization_results.append({
                        'window': window,
                        'step': step,
                        'adjusted_window': 0,
                        'pb_hit_probability': 0,
                        'total_predictions': 0,
                        'total_pb_hits': 0,
                        'performance_ratio': 0,
                        'compute_time': 0,
                        'summary_count': 0,
                        'error': str(e)
                    })
        
        # 排序结果
        optimization_results.sort(key=lambda x: x['pb_hit_probability'], reverse=True)
        
        result_data = {
            'optimization_results': optimization_results,
            'best_params': best_params,
            'best_theoretical_params': best_theoretical_params,
            'total_combinations': total_combinations,
            'data_rows': len(la.df),
            'parameters': {
                'window_range': f"{window_start}-{window_end} (步长{window_step})",
                'step_range': f"{step_start}-{step_end} (步长{step_step})",
                'n_tickets': n_tickets
            }
        }
        
        return render_template('parameter_optimization.html', result=result_data)
        
    except Exception as e:
        return render_template('parameter_optimization.html', error=f"错误：{str(e)}")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)