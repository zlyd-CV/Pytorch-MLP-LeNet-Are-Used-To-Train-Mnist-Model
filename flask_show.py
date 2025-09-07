from flask import Flask, request, jsonify
import torch  # 导入PyTorch
from models import CNN  # 导入你的CNN模型类（必须与保存时的类定义一致）

app = Flask(__name__)

# 关键修复：用torch.load()加载PyTorch模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("models/CNN_MNIST-50.pkl", map_location=device, weights_only=False)
model.to(device)
model.eval()  # 设置为评估模式


@app.route("/predict", methods=["POST"])
def predict():
    # 获取输入数据并预处理（需与训练时的输入格式一致）
    data = request.json["data"]  # 假设输入是图像的像素数据
    # 将数据转换为PyTorch张量，并添加批次维度和通道维度
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 28, 28]
    input_tensor = input_tensor.to(device)

    # 模型预测（关闭梯度计算）
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()  # 获取预测的类别（0-9）

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)