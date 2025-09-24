import pandas as pd
import requests
from io import StringIO
import numpy as np

def get_latest_mega_millions_data():
    """
    获取最新一期 Mega Millions 的开奖号码。
    """
    csv_url = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
    print("正在从开放数据平台下载数据...")
    try:
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        data_string = StringIO(response.text)
        df = pd.read_csv(data_string, nrows=1)
        return df
    except requests.exceptions.RequestException as e:
        print(f"下载数据时出错: {e}")
        return None

def lorenz_system(x, y, z, sigma=10, rho=28, beta=2.667):
    """
    洛伦兹系统的微分方程。
    """
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot

def calculate_lorenz_trajectory(df):
    """
    使用彩票号码作为初始值，计算洛伦兹系统的轨迹。
    """
    if df.empty:
        print("未获取到最新的开奖数据。")
        return
        
    latest_draw = df.iloc[0]
    
    # 获取所有号码，包括Mega Ball
    winning_numbers_str = latest_draw['Winning Numbers'].split()
    all_numbers = [int(n) for n in winning_numbers_str]
    all_numbers.append(int(latest_draw['Mega Ball']))
    
    print(f"最新一期开奖日期: {latest_draw['Draw Date']}")
    print(f"用于计算的数据点: {all_numbers}")
    
    # 将彩票号码映射为洛伦兹系统的初始值
    # 简单的映射：前两个号码之和 -> x0, 中间两个 -> y0, 最后两个 -> z0
    x0 = all_numbers[0] + all_numbers[1]
    y0 = all_numbers[2] + all_numbers[3]
    z0 = all_numbers[4] + all_numbers[5]
    
    # 洛伦兹系统的参数
    sigma, rho, beta = 10, 28, 2.667
    
    # 模拟参数
    dt = 0.01  # 时间步长
    num_steps = 10000 # 模拟步数
    
    # 初始化轨迹列表
    trajectory = []
    
    x, y, z = x0, y0, z0
    
    # 模拟洛伦兹系统
    for _ in range(num_steps):
        trajectory.append((x, y, z))
        dx, dy, dz = lorenz_system(x, y, z, sigma, rho, beta)
        x += dx * dt
        y += dy * dt
        z += dz * dt
    
    print(f"\n--- 洛伦兹系统轨迹计算结果（共 {num_steps} 步） ---")
    print("轨迹上的部分点（前10个）:")
    for point in trajectory[:10]:
        print(f"x: {point[0]:.4f}, y: {point[1]:.4f}, z: {point[2]:.4f}")
    
    print("...")
    
    print("\n注：这些点代表了洛伦兹系统在给定初始值下的演变轨迹。")
    print("它不能用于预测彩票，只是一个有趣的数学和混沌理论的演示。")

def main():
    df = get_latest_mega_millions_data()
    if df is not None:
        calculate_lorenz_trajectory(df)

if __name__ == '__main__':
    main()
