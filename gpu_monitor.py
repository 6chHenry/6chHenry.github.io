import subprocess
import json
import time
from datetime import datetime
import requests
import os

def get_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = result.stdout.strip().split('\n')
        gpu_data = []
        
        for i, line in enumerate(gpu_info):
            if line.strip():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 5:
                    gpu_data.append({
                        'gpu_id': i,
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3]),
                        'power': float(parts[4])
                    })
        
        return gpu_data
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"Error parsing GPU data: {e}")
        return []

def send_feishu_message(webhook_url, message):
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "msg_type": "text",
        "content": {
            "text": message
        }
    }
    
    try:
        response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print(f"Failed to send message: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending message: {e}")

def format_gpu_message(gpu_data):
    if not gpu_data:
        return "❌ 无法获取GPU信息"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🖥️ GPU监控报告 - {timestamp}\n\n"
    
    for gpu in gpu_data:
        memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        message += f"GPU {gpu['gpu_id']}:\n"
        message += f"  使用率: {gpu['utilization']}%\n"
        message += f"  显存: {gpu['memory_used']}/{gpu['memory_total']} MB ({memory_percent:.1f}%)\n"
        message += f"  温度: {gpu['temperature']}°C\n"
        message += f"  功耗: {gpu['power']}W\n\n"
    
    return message

def main():
    os.environ['FEISHU_WEBHOOK_URL'] = 'https://open.feishu.cn/open-apis/bot/v2/hook/6a10d50a-cf31-4229-91af-212ffa5531e5'
    webhook_url = os.getenv('FEISHU_WEBHOOK_URL')
    if not webhook_url:
        print("请设置环境变量 FEISHU_WEBHOOK_URL")
        return
    
    print("GPU监控启动，每5秒检测一次...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            gpu_data = get_gpu_usage()
            message = format_gpu_message(gpu_data)
            
            print(f"发送GPU使用情况: {datetime.now()}")
            send_feishu_message(webhook_url, message)
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
    except Exception as e:
        print(f"监控出错: {e}")

if __name__ == "__main__":
    main()