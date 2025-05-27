from onnx import helper, TensorProto
import onnx

# 加载模型
model = onnx.load(
    "mmdeploy_models/mmaction/uniformerv2/ort/without_dropout.onnx")

# 查找名为 /Softmax 的节点，并找到它的第一个输出
softmax_node = None
for node in model.graph.node:
    if node.op_type == "Softmax" and node.name == "/Softmax":
        softmax_node = node
        break

if softmax_node is None:
    raise RuntimeError("未找到名为 /Softmax 的节点")

# 找到 /Softmax 节点的下一个节点
next_node = None
for i, node in enumerate(model.graph.node):
    if node == softmax_node and i + 1 < len(model.graph.node):
        next_node = model.graph.node[i + 1]
        break
if next_node is None:
    raise RuntimeError("未找到 /Softmax 节点的下一个节点")

# 清空原有输出
model.graph.ClearField("output")

# 删除next_node
model.graph.node.remove(next_node)


# 用 /Softmax 节点的第一个输出作为新输出

new_output = helper.make_tensor_value_info(
    softmax_node.output[0],
    TensorProto.FLOAT,  # 你可以根据实际类型调整
    (1, 4, 2)                # 如果已知 shape 可填写 shape
)
model.graph.output.extend([new_output])


# 保存新模型
onnx.save(model, "mmdeploy_models/mmaction/uniformerv2/ort/uniformerv2_softmax.onnx")
print("已将 /Softmax 节点作为唯一输出")
