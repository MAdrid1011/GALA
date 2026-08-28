# 模型适配与真实数据

## 1. 适配顺序

模型适配器调用作者入口并保留上游训练配置、初始化、优化器、增密、分裂、剪枝和终止条件。适配器只增加 CLAMP 任务映射、设备 trace 写入和结果清单，不替换原模型的数值公式。

| 顺序 | 模型 | 上游仓库 | 固定提交 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | R²-Gaussian | `https://github.com/Ruyi-Zha/r2_gaussian.git` | `f2579bf` | 必须实现 |
| 2 | FaCT-GS | `https://github.com/PaPieta/fact-gs.git` | `9b95ea9` | 必须实现 |
| 3 | Exact-GS | `https://github.com/brucee1323/Exact-GS.git` | `c8f9251` | 必须实现 |
| 4 | GR-Gaussian | 无公开作者源码 | 不适用 | 自动跳过 |

每个适配器保存上游提交、补丁哈希、Python 环境、CUDA 编译参数和官方命令。未启用 trace 时，适配器输出必须与固定提交一致。GR-Gaussian 在获得可核验作者源码前不得用重写实现生成正式结果。

## 2. 数据来源

| 数据名 | 来源 | 输入性质 | 首次用途 |
| --- | --- | --- | --- |
| Chest | R²-Gaussian 发布格式，原始体来自 LIDC-IDRI `LIDC-IDRI-0001` | 官方流程生成的投影、几何与参考体 | 首个闭环组合 |
| Walnut | FIPS 3D cone-beam computed tomography dataset of a walnut | 实测投影、几何与参考重建 | 第二个数据组合 |
| HDTomo-USB | Zenodo 记录 `4822516` | 实测 TXRM 投影、扫描元数据与 USB 参考体 | 第三个数据组合 |

正式数据只能由数据清单中的来源下载。Chest 使用作者公开的数据包时，必须同时记录其生成配置和原始 LIDC-IDRI 标识。禁止用随机体、随机射线、缩小关系集合或自行生成的替代投影形成正式结果。

## 3. 数据清单

每个数据集由 `configs/datasets/<name>.yaml` 描述。清单至少包含下载地址、许可地址、原始文件 SHA-256、解压后关键文件 SHA-256、坐标约定、体素尺寸、探测器尺寸、角度单位、DSO、DSD、训练与测试视角列表、强度预处理、参考体数据范围和官方转换脚本版本。

下载工具只接受清单中的地址，并按以下顺序执行。

1. 下载到内容寻址缓存并校验原始文件。
2. 保存许可、来源时间和来源地址。
3. 使用固定转换器生成模型输入。
4. 校验投影数量、数组形状、数值范围和几何字段。
5. 生成不可变的数据清单哈希。

任何校验失败都将组合标记为 `invalid_dataset`。缺少公开下载或许可时使用 `unavailable_data` 或 `license_blocked`，随后继续下一个可获得组合。

## 4. 适配器合同

每个模型适配器提供以下接口。

```python
class ModelAdapter(Protocol):
    def prepare(self, dataset: DatasetManifest, config: ModelConfig) -> PreparedRun: ...
    def run_reference(self, run: PreparedRun) -> ReferenceArtifact: ...
    def capture_trace(self, run: PreparedRun, sink: DeviceTraceSink) -> TraceArtifact: ...
    def replay_reductions(self, run: PreparedRun, order: ReductionOrder) -> Reconstruction: ...
```

`prepare` 解析官方几何和训练配置。`run_reference` 执行作者路径并生成软件基线。`capture_trace` 使用相同输入、种子和迭代数生成动态事件。`replay_reductions` 只改变合法归约次序，用于核对周期调度下的最终质量。

适配器还必须导出模型阶段边界、每阶段 GPU kernel 列表、数据读写字节数和事件计数。这些信息保留
本地 GPU 阶段证据；只有另有同套件 AGX Orin 实测向量时才用于分阶段平台换算。缺少该实测时
Orin 结果为 `unavailable`，不构成模型适配或 ASIC 主线的退出条件。

## 5. 单组合扩展门

首个正式组合固定为 R²-Gaussian 与 Chest。该组合完成官方质量、trace 完整性、Base ASIC 周期、两类技术上界、两类技术实现、联合结果和全部消融后，才允许适配第二个组合。后续组合不得重新调整冻结的硬件参数。
