import { app } from "../../scripts/app.js";

const labels = {
    source_image: "源图", reference_image: "色彩参考图",
    deband_mode: "去色带模式", manual_strength: "手动去色带强度",
    backend: "计算设备", cpu_workers: "CPU线程上限",
    structure_field: "去色带区域算法", image: "结果图像", report: "处理报告",
    strength: "调色强度",
};
const optionLabels = {
    deband_mode: { auto: "自动", off: "关闭", manual: "手动" },
    backend: { auto: "自动（GPU优先）", cpu: "CPU", gpu: "GPU（强制）" },
    structure_field: { gradient_energy: "梯度能量", dog_guided: "DoG 引导滤波", dog_raw: "原始 DoG" },
};

function localize(node) {
    if ((node.comfyClass || node.type) !== "VelumColorTransfer") return;
    for (const widget of node.widgets || []) {
        if (labels[widget.name]) widget.label = labels[widget.name];
        if (optionLabels[widget.name]) {
            const names = optionLabels[widget.name];
            // Native ComboWidget maps labels in both drawing and menus; values stay English.
            widget.options.getOptionLabel = (value) => names[value] ?? value ?? "";
        }
    }
    for (const slot of [...(node.inputs || []), ...(node.outputs || [])]) {
        if (labels[slot.name]) {
            slot.label = labels[slot.name];
            slot.localized_name = labels[slot.name];
        }
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Velum.ChineseParameters.v2",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VelumColorTransfer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            localize(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            localize(this);
            return result;
        };
    },
    nodeCreated: localize,
    loadedGraphNode: localize,
});
