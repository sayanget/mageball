#!/usr/bin/env python3
import pandas as pd
import numpy as np
import requests
import io

# Mega Millions 数据加载测试
WHITE_MAX = 70
PB_MAX = 25

def load_mega_data(url='https://data.ny.gov/api/views/5xaw-6ayf/rows.csv'):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print(f"从 {url} 加载数据：{len(df)} 行")
        print(f"原始列名：{df.columns.tolist()}")
        print(df.head())
    except Exception as e:
        print(f"下载数据失败：{e}")
        return pd.DataFrame()  # 返回空 DataFrame

    try:
        # 解析日期
        if 'Draw Date' in df.columns:
            df['date'] = pd.to_datetime(df['Draw Date'], format='%m/%d/%Y', errors='coerce')
            invalid_dates = df['date'].isna().sum()
            print(f"无效日期行数：{invalid_dates}")
            if invalid_dates > 0:
                print(f"无效日期示例：{df['Draw Date'][df['date'].isna()].head().tolist()}")
        else:
            df['date'] = pd.NaT
            print("警告：未找到日期列，禁用日期过滤")

        # 解析 Winning Numbers
        if 'Winning Numbers' in df.columns:
            def parse_row(s):
                parts = str(s).split()
                if len(parts) >= 6:
                    return parts[:5] + [parts[5]]
                return [np.nan]*6

            parsed = df['Winning Numbers'].astype(str).apply(parse_row).apply(pd.Series)
            parsed.columns = ['n1','n2','n3','n4','n5','megaball']
            df = pd.concat([df, parsed], axis=1)
        else:
            print("警告：未找到 Winning Numbers 列，使用现有列")
    except Exception as e:
        print(f"解析数据失败：{e}")
        return pd.DataFrame()  # 返回空 DataFrame

    try:
        # 处理 NaN
        df = df.dropna(subset=['date', 'megaball'])
        for c in ['n1','n2','n3','n4','n5','megaball']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('Int64')

        # 日期过滤
        if not df['date'].isna().all():
            initial_rows = len(df)
            df = df[df['date'] >= '2017-10-31']
            print(f"过滤 2017-10-31 后数据：{len(df)} 行（移除了 {initial_rows - len(df)} 行）")
        else:
            print("警告：所有日期无效，跳过过滤")

        print(f"最终有效数据：{len(df)} 行")
        print(df.head())
        return df
    except Exception as e:
        print(f"数据处理失败：{e}")
        return pd.DataFrame()  # 返回空 DataFrame

if __name__ == '__main__':
    df = load_mega_data()
    print("测试完成！")
    if not df.empty:
        print("数据加载成功！")
    else:
        print("数据加载失败，请检查网络或 CSV 格式。")