# Mega Millions 彩票分析器

一个基于Flask的Mega Millions彩票号码分析和预测系统，集成了历史预测记录与回测功能。

## 功能特性

- 🎯 **智能号码分析**: 基于频率、近期表现、间隔时间等多维度分析
- 📊 **历史回测**: 模拟历史数据验证预测策略效果
- 📝 **预测记录**: 自动保存每次预测结果，支持历史回测分析
- 🔄 **实时数据**: 自动从纽约州开放数据平台获取最新开奖信息
- 📈 **可视化图表**: 直观展示分析结果和统计数据
- ⏰ **开奖倒计时**: 实时显示下期开奖时间倒计时

## 技术栈

- **后端**: Flask (Python)
- **数据处理**: Pandas, NumPy
- **前端**: HTML, Tailwind CSS, Chart.js
- **部署**: Render云平台

## 本地运行

1. 克隆仓库
```bash
git clone https://github.com/sayanget/mageball.git
cd mageball
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
python app.py
```

4. 访问 http://localhost:5000

## Render部署

本项目已配置了Render部署文件：
- `requirements.txt`: Python依赖
- `Procfile`: 进程配置
- `runtime.txt`: Python版本
- `render.yaml`: Render服务配置

### 部署步骤

1. 在 [Render](https://render.com) 创建账户
2. 连接GitHub仓库
3. 选择Web Service
4. 配置构建和启动命令（已在配置文件中定义）
5. 部署完成

## 项目结构

```
mageball/
├── app.py                 # 主应用文件
├── templates/             # HTML模板
│   ├── index.html         # 主页面
│   └── prediction_history.html  # 历史预测页面
├── uploads/               # 上传和缓存目录
├── requirements.txt       # Python依赖
├── Procfile              # Render进程配置
├── runtime.txt           # Python运行时版本
├── render.yaml           # Render服务配置
└── README.md             # 项目说明
```

## 核心功能

### 1. 号码分析算法
- **频率分析**: 统计历史出现频率
- **近期趋势**: 分析最近期数的号码表现
- **间隔分析**: 计算号码的出现间隔
- **回测性能**: 基于历史数据的策略验证

### 2. 预测策略
- **Hot策略**: 基于高频号码
- **Cold策略**: 基于低频号码  
- **Hybrid策略**: 综合多种因素的混合策略
- **Random策略**: 完全随机生成

### 3. 历史预测回测
- 自动保存每次预测记录
- 与实际开奖结果对比分析
- 生成详细的性能统计报告
- 支持按策略分组的效果评估

## 数据源

- **Mega Millions数据**: 纽约州开放数据平台
- **数据范围**: 2017年10月31日至今
- **更新频率**: 实时获取最新开奖信息

## 注意事项

⚠️ **重要声明**: 本系统仅供娱乐和学习目的，不构成任何投资建议。彩票是概率游戏，请理性对待。

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目！