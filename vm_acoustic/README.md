# VM-Acoustic paired-microphone pilot

这是已经运行的最小六参数声学探针，不是按元音对齐的24维模型，也不是已校准的人类感知音色地图。

## 实际数据与范围

- 官方 VCTK 0.92，p225–p232，计划001–008，实际62句/124条双通道音频；p227_007、p230_004在官方ZIP选定子集中缺失。
- 同一 utterance ID 的文本已核对一致。对每条音频的共同有效有声帧取 log F0/F1/F2/F3、CPP、HNR 中位数。
- 标定只使用p225–p228的001–004，共16对；其余人/句子分组报告，未参与偏差/尺度/权重拟合。
- 来源：[Edinburgh VCTK](https://datashare.ed.ac.uk/handle/10283/3443)。作者：Junichi Yamagishi、Christophe Veaux、Kirsten MacDonald；CSTR，University of Edinburgh，2019。官方README说明CC BY 4.0；原文保存在data/README.txt及license_text.txt。
- `fetch_samples.py` 从官方 DSpace API 找到ZIP，再严格通过HTTP206范围读取。总共接收43,558,859字节，未下载11.7GB完整包。每个文件经ZIP CRC验证，manifest保留SHA256；未验证整个ZIP的哈希。

## 运行

需要Python 3.12（本机使用应用捆绑的3.12.14）。不必安装torch或任何神经网络。

```powershell
python -m pip install --target vm_acoustic/.deps -r vm_acoustic/requirements.txt
python vm_acoustic/fetch_samples.py
python vm_acoustic/run_probe.py
python vm_acoustic/validate_probe.py
python vm_acoustic/run_probe.py
```

最后一次命令复用特征缓存，把独立检查摘要写入report。全部参数和Praat版本见results/config.json；帧级测量与有效性保存在feature_cache及features.json。修改acoustics.py会改变缓存签名，避免误用旧特征。

## 文件职责

- fetch_samples.py：有界下载、来源元数据与CRC/SHA256记录；不改变音频。
- acoustics.py：六参数、有效帧筛选、仅标定集拟合、固定坐标与距离。
- run_probe.py：划分、提取缓存、校准/留出对比、CSV/JSON报告与本地试听图。
- validate_probe.py：已知合成信号、增益/噪声干预、几何检查与共振峰设置敏感性；不据测试结果调整模型。

## 阅读结果

- results/report.md：主要结果、失败点和解释。
- results/paired_distances.csv：每个配对样本的三种处理距离。
- results/profile_distances.csv：基于005–008的整句声学档案距离，仅作探索。
- results/calibration.json：拟合文件清单、偏差、尺度和平均归一化为1的权重。
- results/validation.json：已知参数与敏感性检查。
- results/map.html：可在支持FLAC的浏览器本地查看/试听。二维图仅用于检查，不是已验证产品。

校准并非所有留出条件均改善。固定5500Hz共振峰上限、未做音素对齐和样本很少，是当前主要限制。身份top1只是同句双麦诊断，不能作为音色相似准确率。

## 元音对齐对照（后续实验）

在原实验之后运行（需安装更新后的requirements）：

```powershell
python vm_acoustic/vowel_probe.py
python vm_acoustic/vowel_profiles.py
python vm_acoustic/validate_vowels.py
```

`vowel_probe.py` 用PocketSphinx默认美式英文HMM对已知文本做两遍对齐，保存每麦音素边界；使用元音中央60%和至少3帧，匹配同词/同音素的两麦token。标定仅使用原来的人/句划分，每元音至少4对。`vowel_profiles.py` 再以句内同元音中位参数做独立对照，要求标定覆盖至少2人。`validate_vowels.py` 检查划分和几何前提，并诊断边界±10ms敏感性。

结果在 `results_vowels/`，不覆盖 `results/`：`report.md` 是token实验，`profile_report.md` 是句内同元音汇总；JSON/CSV保留边界、失败、参数、拟合集及距离。四条录音对齐失败被排除，剩下的自动边界也尚未人工核验。跨方法比较优先读公共尺度raw距离；各自标准化后的距离数值不能直接比较。

当前元音对齐**没有提高公共尺度的双麦重复性**。它帮助定位问题，但短片段噪声、对齐误差和共振峰敏感性仍需解决。训练与AutoDL建议见 `TRAINING_PLAN.md`。

## 测量可靠性排错

```powershell
python vm_acoustic/reliability_probe.py
python vm_acoustic/inspect_reliability.py
```

新增结果在 `results_reliability/`。第一条命令比较96条有已知极点参数的合成音频，依合成开发集选择Burg配置；随后对AE/EH/IY做双麦波形配准、固定60ms相对时间网格、去两端均值和mic1质量拒绝。第二条命令生成真实谱图/轨迹、已知边界干预及检查。`boundary_tracks.png` 已由助手查看，不等同人工语音学标注。`synthetic_examples/` 保存三条有精确边界的合成示例。

原Burg设置在预定合成开发指标中胜出，因此本轮没有更换底层共振峰算法。固定核心在部分条件改善旧元音方法，但配准并非全面改善；26/288候选token通过，覆盖率很低。最终报告列出共同子集的旧元音/整句基线及校准的有效分母，不能把只有1句的校准结果与5句总体比较。流程仍属探索性排错，需要新数据验收。

语言种类与下一步多语言验证设计见 `LANGUAGE_EFFECTS.md`；当前实录实验只覆盖英文。
