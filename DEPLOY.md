# Mageball 彩票分析器 - Render 部署配置

## 📋 部署配置文件说明

### 核心配置文件

1. **render.yaml** - 主要部署配置
   - 服务名称: `mageball-analyzer`
   - Python 3.11运行时
   - 使用自定义启动脚本 `start.py`
   - 端口: 10000（Render默认）

2. **requirements.txt** - Python依赖
   ```
   Flask==3.0.0
   Flask-SocketIO==5.5.1
   pandas==2.2.0
   numpy==1.26.0
   requests==2.31.0
   gunicorn==21.2.0
   eventlet==0.33.3
   ```

3. **start.py** - 生产环境启动脚本
   - 优化的Flask-SocketIO启动配置
   - 环境变量处理
   - 错误处理和日志

4. **build.sh** - 构建脚本
   - 依赖安装
   - 目录创建
   - 权限设置
   - 依赖验证

### 📦 部署步骤

1. **推送代码到GitHub**
   ```bash
   git add .
   git commit -m "更新Render部署配置"
   git push origin main
   ```

2. **在Render中配置**
   - 选择GitHub仓库: `mageball`
   - 环境: Web Service
   - 构建命令: `./build.sh`
   - 启动命令: `python start.py`

3. **环境变量设置**
   - `FLASK_ENV=production`
   - `PYTHON_VERSION=3.11`
   - `PORT=10000` (Render自动设置)

### 🚀 主要功能

- **实时参数优化分析** - 使用SocketIO进度条
- **彩票号码预测** - 基于历史数据分析
- **回测功能** - 验证预测算法效果
- **历史记录** - 保存和分析预测结果

### 🔧 技术栈

- **后端**: Flask + Flask-SocketIO
- **数据处理**: Pandas + NumPy
- **前端**: HTML + TailwindCSS + Chart.js
- **实时通信**: WebSocket (SocketIO)
- **部署**: Render平台

### ⚡ 性能优化

- 使用eventlet异步工作器
- 缓存计算结果
- 优化数据处理流程
- 压缩静态资源

### 🐛 故障排除

1. **依赖安装失败**
   - 检查requirements.txt格式
   - 验证Python版本兼容性

2. **SocketIO连接问题**
   - 确认eventlet正确安装
   - 检查端口配置

3. **内存不足**
   - 优化数据处理窗口大小
   - 减少同时处理的数据量

### 📊 监控和日志

- 健康检查端点: `/health`
- 应用日志自动输出到Render控制台
- 实时性能监控通过Render仪表板

### 🔐 安全考虑

- 生产环境关闭调试模式
- 限制文件上传大小
- API速率限制（如需要）

---

## 部署状态检查

部署完成后，访问以下端点确认服务正常：

- 主页: `https://your-app.onrender.com/`
- 健康检查: `https://your-app.onrender.com/health`
- 预测历史: `https://your-app.onrender.com/prediction_history`
- 参数优化: `https://your-app.onrender.com/parameter_optimization`

预期响应时间: < 5秒（首次冷启动可能较慢）