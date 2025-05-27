import onnx
from onnx import helper, shape_inference


def remove_unconnected_constants(onnx_model):
    # 1. 收集所有被引用的输入名
    used_names = set()
    for node in onnx_model.graph.node:
        for input_name in node.input:
            used_names.add(input_name)
    for out in onnx_model.graph.output:
        used_names.add(out.name)
    for inp in onnx_model.graph.input:
        used_names.add(inp.name)

    # 2. 过滤 initializer，只保留被引用的
    new_initializers = [
        init for init in onnx_model.graph.initializer
        if init.name in used_names
    ]
    onnx_model.graph.ClearField("initializer")
    onnx_model.graph.initializer.extend(new_initializers)

    # 3. 过滤 Constant 节点（只保留有输出被引用的）
    new_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            if any(out in used_names for out in node.output):
                new_nodes.append(node)
        else:
            new_nodes.append(node)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    # 4. 可选：推断形状
    onnx_model = shape_inference.infer_shapes(onnx_model)
    return onnx_model


# 用法示例
if __name__ == "__main__":
    input_onnx_path = "mmdeploy_models/mmaction/uniformerv2/ort/uniformerv2_no_dropout.onnx"
    output_onnx_path = "mmdeploy_models/mmaction/uniformerv2/ort/end2end_no_unconnected_constant.onnx"
    model = onnx.load(input_onnx_path)
    cleaned_model = remove_unconnected_constants(model)
    onnx.save(cleaned_model, output_onnx_path)
    print(f"已删除未连接的Constant，新模型保存至: {output_onnx_path}")
