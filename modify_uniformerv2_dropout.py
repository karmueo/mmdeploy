import onnx
from onnx import helper

# 加载模型
model = onnx.load("mmdeploy_models/mmaction/uniformerv2/ort/end2end.onnx")

# 创建新节点列表
new_nodes = []
for node in model.graph.node:
    if node.op_type == "Dropout":
        # 创建新的Identity节点替代Dropout
        new_node = helper.make_node(
            'Identity',
            inputs=[node.input[0]],  # 只保留data输入
            outputs=[node.output[0]],  # 只保留主输出
            name=node.name
        )
        new_nodes.append(new_node)
    else:
        new_nodes.append(node)

# 更新模型节点
model.graph.ClearField("node")
model.graph.node.extend(new_nodes)

# 清理无用的初始值(initializers)
used_inputs = set()
for node in model.graph.node:
    for input_name in node.input:
        used_inputs.add(input_name)

new_initializers = [
    init for init in model.graph.initializer
    if init.name in used_inputs
]
model.graph.ClearField("initializer")
model.graph.initializer.extend(new_initializers)

# 保存修改后的模型
onnx.save(model, "mmdeploy_models/mmaction/uniformerv2/ort/without_dropout.onnx")
