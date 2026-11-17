# AutoDL：感知坐标实验的可复现训练

本轮已在Windows CPU执行真实JVS感知监督训练、冻结测试、单音频定位和恢复检查。已在 AutoDL RTX 5090 上用 440 条原始录音完成全流程；已测环境和对照结果见 [CLOUD_RESULTS.md](CLOUD_RESULTS.md)。当前不是商用模型。

## 环境

Python3.12，工作目录为仓库根目录。为本轮小型坐标头租大GPU没有必要；冻结前端提取耗时更大，才可考虑GPU。驱动/CUDA匹配按[PyTorch官方版本表](https://pytorch.org/get-started/previous-versions/)和[AutoDL依赖说明](https://www.autodl.com/docs/deps/)选择，未做显存实测。

```bash
# CPU可完整复现；选择CUDA时把本行换为官方匹配构建，torch/torchaudio版本保持一致
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r vm_perceptual/requirements.txt
python vm_perceptual/test_pipeline.py
```

在云端正常安装，不复制Windows的.deps或DLL。SpeechBrain/Transformers权重与数据均单独获取，源码包不携带。

## 从零准备数据

先阅读[官方JVS使用条款](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)。此脚本按非商业研究范围使用，不代表产品授权。源码公开与语音再分发是不同事项。

```bash
python vm_perceptual/prepare_sources.py
python vm_perceptual/fetch_jvs.py
python vm_perceptual/fetch_models.py --output vm_perceptual/models
python vm_perceptual/extract_features.py --models vm_perceptual/models --device cpu
```

固定下载444项（440段音频和4项元数据），实际范围读取109.91MB，另有5.46MB ZIP目录；原ZIP3.5GB不整体下载。脚本验证精确206范围、成员CRC、音频SHA256及评分文件固定commit/hash。网络失败可重跑，已有音频按CRC复用；特征缓存按源码/音频哈希失效。

前端GPU可将最后一行device改为cuda。下载过权重后可设置HF_HUB_OFFLINE=1。

## 训练与冻结评估

```bash
python vm_perceptual/train_map.py --device cpu --epochs 300
# 确认开发选择与基线已固定后，只执行一次最终测试
python vm_perceptual/evaluate_map.py --confirm-heldout
python vm_perceptual/short_clip_check.py
```

60训练/20开发/20测试人，按性别分层；跨划分配对不用于拟合。训练12个预定候选，开发集选择包含不训练基线，测试不参与选择。主模型只有一个32→16线性投影加2个评分标定参数（514参数），非从零训练大模型。所有候选研究结果均报告，不按测试结果替换选定模型。

已有目录拒绝静默覆盖。--resume增加指定轮数，校验数据哈希并恢复优化器/历史：

```bash
python vm_perceptual/train_map.py --representations acoustic --dimensions 16 --epochs 50 --resume
```

这会改变训练状态，只应在未打开最终测试前或另立新实验时使用。不要把反复查看相同测试集称为新的盲测。results/evaluation.json存在时最终评估入口拒绝重复运行。

## 单音频定位

```bash
python vm_perceptual/infer.py /path/to/mono.wav \
  --checkpoint vm_perceptual/runs/selected.pt \
  --projection vm_perceptual/results/projection.json \
  --models vm_perceptual/models \
  --display-dimensions 2 --output query.json
```

输出真实的16维研究坐标、可选1/2/3维投影及4个未训练声学语气描述量。checkpoint与投影哈希必须匹配。不是已验证的语气嵌入；基准化和身份关联明确返回not_implemented。任意音频的不确定区间尚未校准，输出null并说明，不能当作零误差。当前选出的声学模型推理不加载ECAPA或WavLM权重。

保存runs、features及来源清单到私有研究数据目录并备份；不得把原始JVS/ESD语音推入GitHub。此实验未使用ESD。数据、训练产物的用途仍受来源条款约束。

## 固定地图与近邻查询

完成上述推理得到 query.json 后：

```bash
python vm_perceptual/map_index.py build
python vm_perceptual/map_index.py search query.json --k 5
```

索引包含每条录音的完整音色向量和前三个显示坐标。检索使用完整向量距离，不使用二维显示距离；默认排除相同音频哈希，避免查询自己的零距离掩盖实际效果。查询与索引必须来自相同模型版本。当前索引仅是研究数据地图，不是已验证的身份库或产品准确率。跨解析器 adapter_check.py 保留为低优先级研究脚本，不是主流程步骤。

源码 ZIP 将历史报告放在 reference_results/。新训练在 results/ 写入本次结果。
