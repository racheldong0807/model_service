import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

# =====================================
# 推理流程讲解
# =====================================

# ------------------------------
# 1. 输入数据（Input）
# - 输入必须与训练时一致：
#   - 图像尺寸：28×28
#   - 通道数：1（灰度）
#   - 数据类型：float32
#   - shape: [batch, 1, 28, 28]
# ------------------------------
def load_image(image_path):
    img = Image.open(image_path).convert("L")  # 转为灰度
    return img


# ------------------------------
# 2. 预处理（Preprocessing）
# - 与训练阶段保持一致：
#   - 转 tensor
#   - 归一化
#   - 批处理
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess(img):
    tensor = transform(img).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor.numpy().astype(np.float32)


# ------------------------------
# 3. 模型推理（Forward）
# - 使用 ONNX Runtime 进行推理
# - 注意：ONNX 没有训练模式概念，默认就是推理
# ------------------------------
def run_inference(onnx_model_path, input_data):
    # 创建 ONNX Runtime 推理会话
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # 准备输入
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}

    # 前向推理
    ort_outs = ort_session.run(None, ort_inputs)  # 输出 logits（未归一化分数）
    return ort_outs[0]


# ------------------------------
# 4. 后处理（Postprocessing）
# - Softmax → 概率
# - argmax → 预测标签
# ------------------------------
def postprocess(logits):
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    pred_label = np.argmax(probs, axis=1)[0]
    pred_conf = np.max(probs, axis=1)[0]
    return pred_label, pred_conf, probs


# ------------------------------
# 5. 返回结果（Output）
# - 封装预测结果与概率
# - 可用于 API 返回或前端展示
# ------------------------------
def predict(image_path, model_path="model/mnist_cnn.onnx"):
    img = load_image(image_path)                      # Step 1 输入数据
    input_data = preprocess(img)                      # Step 2 预处理
    logits = run_inference(model_path, input_data)    # Step 3 模型推理
    pred_label, pred_conf, probs = postprocess(logits) # Step 4 后处理

    result = {
        "预测结果": int(pred_label),
        "置信度": float(pred_conf),
        "概率分布": probs.tolist()
    }
    return result


if __name__ == "__main__":
    # 你可以放一张 MNIST 手写数字图片，比如 test.png
    result = predict("test.png")
    print("✅ 推理结果：", result)
