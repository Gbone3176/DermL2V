import numpy as np  
from numba import cuda  
import time  
import os  
import argparse  

def setup_args():  
    """  
    设置命令行参数  
    """  
    parser = argparse.ArgumentParser(description='GPU 任务执行器')  
    parser.add_argument('--gpu', type=int, default=1,  
                      help='指定要使用的 GPU ID（默认：0）')  
    parser.add_argument('--memory', type=float, default=21,  
                      help='指定要使用的显存大小（GB）（默认：2GB）')  
    parser.add_argument('--utilization', type=int, default=45,  
                      help='目标GPU利用率百分比（默认：30%%）')  
    parser.add_argument('--interval', type=float, default=1.0,  
                      help='任务执行间隔时间（秒）（默认：1.0秒）')
    return parser.parse_args()  

def select_gpu(gpu_id):  
    """  
    选择指定的 GPU  
    """  
    device_count = cuda.gpus.lst  
    if len(device_count) == 0:  
        print("没有可用的 GPU 设备！")  
        exit(1)  

    print(f"系统中有 {len(device_count)} 个可用的 GPU:")  
    for i, device in enumerate(cuda.gpus.lst):  
        print(f"GPU {i}: {device.name}")  

    if gpu_id < 0 or gpu_id >= len(device_count):  
        print(f"无效的 GPU ID: {gpu_id}，必须在 0 到 {len(device_count) - 1} 之间")  
        exit(1)  

    cuda.select_device(gpu_id)  
    print(f"已选择 GPU {gpu_id}: {cuda.get_current_device().name}")  

@cuda.jit  
def light_computation(a, b, c):  
    """  
    CUDA 核函数：执行轻量级向量运算  
    """  
    idx = cuda.grid(1)  
    if idx < a.size:  
        # 减少计算循环次数，降低负载
        for _ in range(30):  # 从100减少到10
            c[idx] = a[idx] + b[idx] * 0.5 + idx * 0.001

def calculate_array_size(target_memory_gb):  
    """  
    根据目标显存计算数组大小  
    """  
    bytes_per_element = 4  
    total_bytes = target_memory_gb * 1024 * 1024 * 1024  
    array_bytes = total_bytes // 3.5  
    return int(array_bytes // bytes_per_element)  

def calculate_work_time(target_utilization, interval):
    """
    根据目标利用率计算工作时间和休息时间
    """
    work_time = (target_utilization / 100.0) * interval
    sleep_time = interval - work_time
    return max(0.1, work_time), max(0.1, sleep_time)  # 最小0.1秒

def gpu_task(memory_gb, target_utilization, interval):  
    """  
    执行轻量级 GPU 任务  
    """  
    # 根据指定的显存大小计算数组大小（使用更小的数组）
    N = min(calculate_array_size(memory_gb), 1500000000)  # 限制最大数组大小
    actual_memory = (N * 4 * 3) / (1024**3)  # 计算实际使用的显存
    print(f"将使用约 {actual_memory:.2f}GB 显存（每个数组大小：{N} 个元素）")
    print(f"目标GPU利用率：{target_utilization}%，任务间隔：{interval}秒")

    # 计算工作时间和休息时间
    work_time, sleep_time = calculate_work_time(target_utilization, interval)
    print(f"每个周期：工作 {work_time:.2f}秒，休息 {sleep_time:.2f}秒")

    # 初始化较小的数据集
    print("正在初始化数据...")  
    a = np.random.random(N).astype(np.float32)
    b = np.random.random(N).astype(np.float32)  
    c = np.zeros_like(a)  

    # 将数据传输到 GPU  
    print("正在将数据传输到 GPU...")  
    d_a = cuda.to_device(a)  
    d_b = cuda.to_device(b)  
    d_c = cuda.to_device(c)  

    # 配置较小的线程块
    threads_per_block = 128  # 从256减少到128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block  

    print(f"\n开始运行任务（目标利用率：{target_utilization}%），按 Ctrl+C 停止程序...")  
    try:  
        iteration = 0  
        while True:  
            iteration += 1
            
            # 工作阶段
            work_start = time.time()
            while time.time() - work_start < work_time:
                # 调用核函数执行轻量级计算  
                light_computation[blocks_per_grid, threads_per_block](d_a,d_b,d_c)  
                cuda.synchronize()  

            # 休息阶段
            print(f"迭代 {iteration}: 工作了 {work_time:.2f}秒，休息 {sleep_time:.2f}秒")
            time.sleep(sleep_time)

    except KeyboardInterrupt:  
        print("\n检测到 Ctrl+C，正在退出程序...")  

    finally:  
        # 清理 GPU 内存  
        d_a.free()  
        d_b.free()  
        d_c.free()  
        print("已清理 GPU 内存")  

if __name__ == "__main__":  
    os.environ['NUMBA_ENABLE_CUDASIM'] = '0'  

    # 解析命令行参数  
    args = setup_args() 
    print(args.memory)
    # 选择 GPU  
    select_gpu(args.gpu)

    # 执行 GPU 任务  
    gpu_task(args.memory, args.utilization, args.interval)