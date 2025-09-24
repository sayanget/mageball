import pandas as pd
import random
import requests
from io import StringIO
from collections import Counter

def get_mega_millions_data():
    """
    获取 Mega Millions 历史开奖数据。
    """
    csv_url = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
    print("正在从开放数据平台下载数据...")
    try:
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        data_string = StringIO(response.text)
        df = pd.read_csv(data_string)
        return df
    except requests.exceptions.RequestException as e:
        print(f"下载数据时出错: {e}")
        return None

def clean_data(df):
    """
    清洗数据，处理号码格式。
    """
    # 确保列名与实际数据匹配
    df.columns = [
        'Draw Date', 'Winning Numbers', 'Mega Ball', 'Multiplier'
    ]
    
    # 分割中奖号码，并处理可能的空白
    df['Winning Numbers'] = df['Winning Numbers'].astype(str).apply(
        lambda x: [int(n) for n in x.split()]
    )
    return df

def generate_hot_numbers(df):
    """
    生成最常出现的号码组合。
    """
    all_numbers = []
    for numbers in df['Winning Numbers']:
        all_numbers.extend(numbers)
    
    number_counts = Counter(all_numbers)
    # 获取最常出现的5个号码
    hot_numbers = [num for num, count in number_counts.most_common(5)]
    
    mega_ball_counts = Counter(df['Mega Ball'])
    # 获取最常出现的 Mega Ball
    hot_mega_ball = mega_ball_counts.most_common(1)[0][0]
    
    return sorted(hot_numbers), hot_mega_ball

def generate_cold_numbers(df):
    """
    生成最不常出现的号码组合。
    """
    all_numbers = []
    for numbers in df['Winning Numbers']:
        all_numbers.extend(numbers)
    
    # 获取所有可能的普通号码和 Mega Ball 号码
    possible_numbers = set(range(1, 71))
    possible_mega_balls = set(range(1, 26))

    all_drawn_numbers = set(all_numbers)
    all_drawn_mega_balls = set(df['Mega Ball'])

    # 找出未出现的号码
    cold_numbers = list(possible_numbers - all_drawn_numbers)
    cold_mega_balls = list(possible_mega_balls - all_drawn_mega_balls)
    
    # 如果没有未出现的号码，则获取出现次数最少的号码
    if len(cold_numbers) < 5:
        number_counts = Counter(all_numbers)
        sorted_numbers = sorted(number_counts.items(), key=lambda item: item[1])
        cold_numbers = [num for num, count in sorted_numbers[:5]]
    
    if not cold_mega_balls:
        mega_ball_counts = Counter(df['Mega Ball'])
        sorted_mega_balls = sorted(mega_ball_counts.items(), key=lambda item: item[1])
        cold_mega_balls.append(sorted_mega_balls[0][0])
        
    # 从冷号码中随机选择5个
    selected_cold_numbers = random.sample(cold_numbers, 5)
    selected_cold_mega_ball = random.choice(cold_mega_balls)
    
    return sorted(selected_cold_numbers), selected_cold_mega_ball

def generate_random_numbers():
    """
    生成完全随机的号码组合。
    """
    white_balls = sorted(random.sample(range(1, 71), 5))
    mega_ball = random.randint(1, 25)
    return white_balls, mega_ball

def run_backtest(df):
    """
    运行回测，模拟选号策略在历史数据上对 Mega Ball 的表现。
    """
    print("\n--- 正在运行回测（Mega Ball 命中率）---")
    
    # 获取不同策略的选号
    hot_nums, hot_mega_ball = generate_hot_numbers(df)
    cold_nums, cold_mega_ball = generate_cold_numbers(df)
    
    # 统计Mega Ball命中次数
    hot_mega_ball_hits = 0
    cold_mega_ball_hits = 0
    random_mega_ball_hits = 0

    # 遍历所有历史开奖数据
    for index, row in df.iterrows():
        winning_mega_ball = row['Mega Ball']
        
        # 热门号码策略
        if hot_mega_ball == winning_mega_ball:
            hot_mega_ball_hits += 1
        
        # 冷门号码策略
        if cold_mega_ball == winning_mega_ball:
            cold_mega_ball_hits += 1
            
        # 随机号码策略
        random_mega_ball = random.randint(1, 25)
        if random_mega_ball == winning_mega_ball:
            random_mega_ball_hits += 1

    print("\n--- 回测结果（历史数据总计：{}期） ---".format(len(df)))
    print(f"热门号码 Mega Ball 命中次数: {hot_mega_ball_hits} 次")
    print(f"冷门号码 Mega Ball 命中次数: {cold_mega_ball_hits} 次")
    print(f"随机号码 Mega Ball 命中次数: {random_mega_ball_hits} 次")
    
    print("\n注：结果表明，即使只看 Mega Ball，任何策略的命中率依然非常低。")
    print("彩票是一个完全随机的游戏，请理性对待。")

def main():
    df = get_mega_millions_data()
    if df is not None:
        df = clean_data(df)
        
        print("\n--- Mega Millions 号码选择器 ---")
        
        # 生成并打印最常出现的号码
        hot_nums, hot_mega_ball = generate_hot_numbers(df)
        print("\n**最常出现的号码（仅供娱乐）：**")
        print(f"普通号码: {hot_nums}")
        print(f"Mega Ball: {hot_mega_ball}")
        
        # 生成并打印最不常出现的号码
        cold_nums, cold_mega_ball = generate_cold_numbers(df)
        print("\n**最不常出现的号码（仅供娱乐）：**")
        print(f"普通号码: {cold_nums}")
        print(f"Mega Ball: {cold_mega_ball}")
        
        # 生成并打印完全随机的号码
        random_nums, random_mega_ball = generate_random_numbers()
        print("\n**完全随机的号码（最公平的选择）：**")
        print(f"普通号码: {random_nums}")
        print(f"Mega Ball: {random_mega_ball}")
        
        # 运行回测
        run_backtest(df)
        
        print("\n**祝你好运！**")

if __name__ == '__main__':
    main()
