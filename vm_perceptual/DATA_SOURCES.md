# 数据与引用

JVS语料：Shinnosuke Takamichi、Ryouta Sonobe、Kentaro Mitsui、Yuki Saito、Tomoki Koriyama、Naoko Tanji、Hiroshi Saruwatari，JSUT and JVS: Free Japanese voice corpora for accelerating speech synthesis research，Acoustical Science and Technology 41(5), 761–768, 2020。

原作者页面与具体使用范围：[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)。本轮使用公开提供的研究用途音频，原语音不再分发，不代表获得商用授权；标签按原页面CC BY-SA 4.0说明处理。保留下载包README。文本来源JSUT，其条款独立。

感知评分：Yuki Saito、Shinnosuke Takamichi、Hiroshi Saruwatari，Perceptual-similarity-aware deep speaker representation learning for multi-speaker generative modeling，IEEE/ACM TASLP 29, 1033–1048, 2021。原作者[评分仓库](https://github.com/sarulab-speech/jvs_similarity_score)，固定commit d709591d262c8c80fe4079ebf87f49c32ce61b10，原始标签不在本源码包内，下载脚本核对hash。

冻结模型：[SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)、[WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)，固定版本见extract_features.py。复用已有本地快照，未修改第三方代码或重新训练这些编码器。使用或再分发权重前保留各自模型卡与许可。

本轮没有使用ESD数据拟合本感知模型，也没有在源码包中包含ESD、JVS音频、原始评分、第三方权重或本研究checkpoint。

英语听感对照使用 Yuki Saito 公开的 62 人 VCTK 女性声线评分矩阵及对应名单。在作者页面的“Research Project 2 / In English”下载 `VCTK_62females.list` 和 `VCTK_62females.csv`，放在 `vm_perceptual/data_sources/`。`english_listener_check.py` 对已有双麦音频与这些声线取交集，核对原始文件哈希和文本；本次共 11 人。该数据没有用于选择训练配方或权重。[原始评分和名单](https://sython.org/demo/JSPS-DC1/index.html)
