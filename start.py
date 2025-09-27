#!/usr/bin/env python3
"""
生产环境启动脚本
用于在Render等平台上启动Flask-SocketIO应用
"""
import os
import sys
from app import app, socketio

if __name__ == '__main__':
    # 设置生产环境变量
    os.environ['FLASK_ENV'] = 'production'
    
    # 获取端口号
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动Mageball分析器在端口 {port}")
    print(f"🌐 环境: {os.environ.get('FLASK_ENV', 'development')}")
    
    try:
        # 使用eventlet异步工作器启动
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)