#!/bin/bash

# Render部署脚本
# 这个脚本会在Render上自动执行

echo "开始安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

echo "创建必要的目录..."
mkdir -p uploads

echo "设置权限..."
chmod 755 uploads

echo "依赖安装完成，准备启动应用..."