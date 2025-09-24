import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_mega_millions_data():
    """
    获取 Mega Millions 历史开奖数据。
    """
    # 更新为更稳定的第三方数据源
    csv_url = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
    print("正在从开放数据平台下载数据...")
    try:
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status() # 检查请求是否成功
        # 使用 StringIO 将文本内容转换为文件对象，供 pandas 读取
        data_string = StringIO(response.text)
        # 修复：移除 header=None 参数，让 pandas 自动使用第一行作为列名
        df = pd.read_csv(data_string)
        return df
    except requests.exceptions.RequestException as e:
        print(f"下载数据时出错: {e}")
        return None

def clean_data(df):
    """
    清洗数据，处理日期和号码格式。
    """
    # 修复：确保列名与实际数据匹配
    # 新的数据源有不同的列名，这里进行相应调整
    df.columns = [
        'Draw Date', 'Winning Numbers', 'Mega Ball', 'Multiplier'
    ]
    df['Draw Date'] = pd.to_datetime(df['Draw Date'], format='%m/%d/%Y')
    
    # 分割中奖号码，并处理可能的空白
    # 注意：新数据源的 Winning Numbers 列是以空格分隔的字符串
    df['Winning Numbers'] = df['Winning Numbers'].astype(str).apply(
        lambda x: [int(n) for n in x.split()]
    )
    return df

def analyze_and_visualize(df):
    """
    分析数据并生成可视化图表。
    """
    all_numbers = []
    for numbers in df['Winning Numbers']:
        all_numbers.extend(numbers)
    
    # 统计每个号码的出现频率
    number_counts = pd.Series(all_numbers).value_counts().sort_index()

    # 统计 Mega Ball 的出现频率
    mega_ball_counts = df['Mega Ball'].value_counts().sort_index()

    # 绘制最常出现的普通号码
    plt.figure(figsize=(15, 6))
    sns.barplot(x=number_counts.index, y=number_counts.values, palette="viridis")
    plt.title('普通号码出现频率', fontsize=16)
    plt.xlabel('号码', fontsize=12)
    plt.ylabel('出现次数', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('regular_numbers_frequency.png')
    plt.show()

    # 绘制最常出现的 Mega Ball
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mega_ball_counts.index, y=mega_ball_counts.values, palette="rocket")
    plt.title('Mega Ball 出现频率', fontsize=16)
    plt.xlabel('号码', fontsize=12)
    plt.ylabel('出现次数', fontsize=12)
    plt.tight_layout()
    plt.savefig('mega_ball_frequency.png')
    plt.show()
    
    # 打印最常出现的号码
    print("\n--- 最常出现的普通号码（前5名）---")
    print(number_counts.nlargest(5))

    print("\n--- 最常出现的 Mega Ball（前5名）---")
    print(mega_ball_counts.nlargest(5))

def main():
    df = get_mega_millions_data()
    if df is not None:
        df = clean_data(df)
        analyze_and_visualize(df)

if __name__ == '__main__':
    main()
