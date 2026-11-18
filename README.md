# VocalMap：数学声学与音色空间研究

在本次 AutoDL 实例的 JupyterLab 中打开 `/root/autodl-tmp/volcalmap/Open_VocalMap.ipynb`，运行全部单元即可听查询录音、查看 16 维音色坐标和前五近邻。修改首个代码单元的 `audio`、`map_variant` 和 `display_dimensions` 可更换音频、比较默认图与实验候选，以及切换一至三维显示。云端已准备本轮使用的录音、权重和独立索引。[演示 Notebook](Open_VocalMap.ipynb) · [比较结果](vm_perceptual/WAVE_RESULTS.md)


[云端音色地图比较与实际查询](vm_perceptual/WAVE_RESULTS.md) · [AutoDL 演示笔记本](Open_VocalMap.ipynb)

最终目标：输入音频，得到感知音色坐标；可选输出跨音色共享的语气坐标。模仿者听起来相似应靠近，同人不同声线可以分开。语气匹配与语气基准化是两个独立控制。

## 最新：真实感知监督的研究地图

已新增[VM-Perceptual](vm_perceptual/README.md)：100人JVS、60/20/20人隔离、冻结前端＋小型感知距离头，完整训练、留出评价及单音频→16维坐标→1/2/3维显示已运行。

开发集选定模型测试听感排序相关0.413，未超过完整声学基线0.518；不能称为已验证的最终模型。短音频漂移明显，跨解析器转换仍有较大误差，语气基准化和同人关联未实现。[部署步骤](vm_perceptual/AUTODL.md)、[完整结果](vm_perceptual/results/REPORT.md)、[坐标标准化](vm_perceptual/COORDINATE_STANDARD.md)。Linux/CUDA/AutoDL尚未实际部署。

下方为此前VM-Acoustic测量阶段记录，不能与新感知指标混用。

## 本轮结果（2026-11-17）

|验证|结果|边界|
|---|---|---|
|同一批410个AE/EH/IY候选的可计算覆盖|旧门控22/410，新方法363/410|88.5%是可计算率，不是准确率|
|独立自然元音参考，成人测试|路径法96.92%的参考点误差≤10%，固定40ms为93.85%|成人测试已查看，属于探索性结果|
|未用于调参的12名儿童72条|91.72%对82.78%|仍未达到95%目标|
|冻结方法，新4人24对双麦录音|距离中位0.3399→0.3057，降低10.0%|按人bootstrap区间跨0；长元音0.2656→0.2700，未改善|
|74,115参数物理预测小网络|测试误差中位4.66%，Burg为1.16%|负面结果，不建议直接扩网络或租大GPU|

全部预定长片段保留。HMM对齐中存在静音与辅音混入，当前没有用删除困难样本制造改进。Hillenbrand数据是自然/hVd/词，不能把它说成完整自然载句。

## 复现与部署

Python 3.12。依赖安装、数据重建、训练、恢复、评估见 [AutoDL手册](vm_acoustic/AUTODL_RUNBOOK.md)。无需大GPU即可复现实验。

```bash
python -m pip install -r vm_acoustic/requirements.txt
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python vm_acoustic/test_contracts.py
python vm_acoustic/train_physics.py --smoke --output vm_acoustic/training_runs/smoke
```

Windows CPU 与 AutoDL 的契约测试已运行。GitHub Actions 已在提交 59675ff 的 Linux/Windows 两个任务通过。不包含原始音频、依赖DLL、模型权重和账户凭据。原音频需按来源条款重新获取，不因代码托管而改变其许可。

## 阅读顺序

1. [研究就绪度和下一步](vm_acoustic/READINESS.md)：工程可运行与正式模型验证分开。
2. [产品与训练目标](vm_acoustic/MODEL_CONTRACT.md)：输入输出、音色优先、语气开关、训练关卡。
3. [独立双麦确认](vm_acoustic/results_confirmation/report.md)、[边界图审查](vm_acoustic/BOUNDARY_REVIEW.md)。
4. 本地 thinking.md：合作实验记录；并非模型的内部推理记录。

旧版六参数探针说明保留在 vm_acoustic/README.md。新测量入口为 measurement_v2.py，路径法为 formant_tracker.py。voice_space.py只验证坐标检索契约，不伪造音频编码器。
