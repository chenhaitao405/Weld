import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import argparse  # 导入命令行参数解析模块

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['AR PL UKai CN', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
print(f"\n已设置字体为: AR PL UKai CN")

from ultralytics import YOLO


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLO11 分割模型训练脚本')

    # 添加 data 参数（必填，指定数据集配置文件）
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径（如 crackseg.yaml）')

    # 可选：添加其他可配置参数（如果需要灵活调整）
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数（默认：500）')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸（默认：320）')
    parser.add_argument('--batch', type=int, default=32,
                        help='批次大小（默认：32）')
    parser.add_argument('--name', type=str, default="crack_11m_nopretrain_lr",
                        help='训练结果保存名称（默认：crack_11m_nopretrain_lr）')
    parser.add_argument('--model_nopretrain_lr）')
    parser.add_argument('--model', type=str, default="yolo11x-seg.yaml",
                        help='模型配置文件路径（默认：yolo11m-seg.yaml）')

    # 解析参数
    args = parser.parse_args()

    # 加载模型
    model = YOLO(args.model)  # 从配置文件构建模型

    # 训练模型（使用命令行传入的参数）
    print(f"\n开始训练，数据集配置：{args.data}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name
    )


if __name__ == "__main__":
    main()
