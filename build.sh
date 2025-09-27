#!/bin/bash

# Render部署脚本
# 这个脚本会在Render上自动执行

echo "开始安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

echo "安装SocketIO支持库..."
pip install python-socketio python-engineio

echo "创建必要的目录..."
mkdir -p uploads
mkdir -p templates

echo "设置权限..."
chmod 755 uploads
chmod 755 templates

echo "验证依赖安装..."
python -c "import flask, flask_socketio, pandas, numpy, requests; print('所有依赖安装成功')"

echo "依赖安装完成，准备启动应用..."