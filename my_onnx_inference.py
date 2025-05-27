import cv2
import numpy as np
import onnxruntime as ort


def load_onnx_model(model_path):
    """加载ONNX模型并创建推理会话"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options)


def load_txt_data(txt_path, target_shape):
    # 读取txt文件（每行一个数值）
    with open(txt_path, 'r') as f:
        data = np.array([float(line.strip()) for line in f], dtype=np.float32)

    # 重塑为模型输入维度（例如 [1,4,3,8,224,224]）
    return data.reshape(target_shape)


def load_bin_data(bin_path, shape=None, dtype=np.float32):
    """
    读取二进制bin文件并转换为指定形状的NumPy数组

    参数:
        bin_path (str): 输入bin文件路径
        shape (tuple): 目标形状（如 (1,4,2)），若为None则保持1D
        dtype (np.dtype): 数据类型，默认为np.float32

    返回:
        np.ndarray: 转换后的数组
    """
    # 读取二进制文件
    data = np.fromfile(bin_path, dtype=dtype)

    # 检查数据量是否匹配目标形状
    if shape is not None:
        expected_size = np.prod(shape)  # 计算目标形状的总元素数
        if len(data) != expected_size:
            raise ValueError(
                f"数据量不匹配: 文件包含 {len(data)} 个元素，但目标形状 {shape} 需要 {expected_size} 个元素"
            )
        data = data.reshape(shape)  # 转换为指定形状

    return data


def preprocess_frame(frame, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """预处理单帧图像（强制输出float32）"""
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)  # 关键修改：明确指定float32
    frame = (frame - np.array(mean, dtype=np.float32)) / \
        np.array(std, dtype=np.float32)
    return np.transpose(frame, (2, 0, 1))  # HWC -> CHW


def extract_video_clip(video_path, clip_length=16):
    """从视频中提取指定长度的片段"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < clip_length:
        ret, frame = cap.read()
        if not ret:
            # 如果视频结束，循环从头开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(frame)

    cap.release()
    return frames[:clip_length]  # 确保长度准确


def build_model_input(frames):
    """构建模型输入张量 [1, 16, 3, 224, 224] (NTCWH)"""
    processed_frames = [preprocess_frame(f) for f in frames]
    # 堆叠帧并添加batch维度
    return np.expand_dims(np.stack(processed_frames, axis=0), axis=0)


def main():
    # 配置参数
    model_path = "mmdeploy_models/mmaction/uniformerv2/ort/uniformerv2_softmax.onnx"  # 替换为你的模型路径
    # video_path = "/home/tl/data/datasets/video/vidoe_recognition_data/1.mp4"  # 替换为你的视频路径

    # 1. 加载模型
    print("Loading ONNX model...")
    sess = load_onnx_model(model_path)
    input_name = sess.get_inputs()[0].name

    # 2. 处理视频
    # print(f"Processing video: {video_path}")
    # frames = extract_video_clip(video_path)
    # input_tensor = build_model_input(frames)
    # 直接从txt文件加载数据
    # input_tensor = load_txt_data(
    # "/home/tl/work/mmdeploy/input_data.txt", [1, 4, 3, 8, 224, 224])
    input_tensor = load_bin_data(
        "output.bin", [1, 4, 3, 8, 224, 224])

    # 3. 执行推理
    print("Running inference...")
    outputs = sess.run(None, {input_name: input_tensor})

    # 4. 输出结果
    print("Inference results:")
    print(outputs)
    print("Inference completed!")


if __name__ == "__main__":
    main()
