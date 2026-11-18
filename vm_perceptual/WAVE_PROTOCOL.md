# 单句音色地图的候选模型与评估

默认地图为已复现的 acoustic_d16。以下实验使用独立目录。权重、语音和听众逐条评分存放在研究环境内。

## 音频与评价单位

JVS 的人工标签描述同性别的两种正常声线。单句实验抽取两人的不同句子，计算地图相似度与声线评分的 Spearman 相关。每种不同文本分配及两个性别等权。四句画像、单句、跨麦同次录音检索和同人真假声变化各有独立指标。所有公开的评分已被研究人员查阅。

## 波形学生

wave_student.py 从单通道语音学习 16 维单位向量。训练采用完整语句的固定声学教师坐标和 1、2、3、4、6 秒随机裁剪。三种配方分别训练三个种子，每次 2,000 步。先按三个种子的平均开发效用选择配方，再选择该配方开发效用居中的种子。效用为完整单句开发相关减去四种时长下平均教师坐标误差的一半。仅对锁定的 wave_431_noise 作旧测试人对照。wave_comparison.py 给出相同录音与时长的原地图和学生结果。

## 外部模型

ECAPA-TDNN 的身份训练提供独立的声纹识别基线。VoxSim 官方 WavLM–ECAPA 提供听感比较基线。voxsim_reference.py 将每条录音的完整句单位向量与十个窗口单位向量的均值拼接并除以平方根二。两个结果的点积等于作者打分公式，两边向量再次归一化会改变该公式。小 ECAPA 公布权重缺少末端 BN，代码以 Identity 重建这一层，名称标记 reconstructed。大模型严格加载参数。权重 SHA-256 在脚本内核验。[VoxSim](https://mm.kaist.ac.kr/projects/voxsim/) · [作者模型与实现](https://github.com/kaistmm/voxsim_trainer)

reference_benchmark.py 在同一组语句比较各模型，四句画像采用句向量均值的点积。报告按性别分层的 1,500 次说话人重采样差值区间，同一个人的配对不会被当作独立观测。wave_style_audit.py 的普通声和假声没有完全相同的文本，结果展示声线变化响应。

## 按人重新训练和多文本确认

wave_folds.py、perceptual_rank.py 与 reference_student.py 均排除原测试人。其余 80 人按性别分五折，轮换训练、开发与外层验证角色，重复五次。训练使用第 1–4 句，开发使用第 5–8 句，外层使用第 9–12 句。每次仅在当折训练人上拟合中心、尺度、降维和教师，开发人选训练步数。重复分组共享人员，不能作为 25 个独立数据集做显著性检验。

- wave_folds.py 比较小型波形学生对声学教师、以及附加 ECAPA 相对几何的学习结果。
- perceptual_rank.py 测试 16 维线性映射对人工相似度次序的学习能力。仅比较同一锚点、听感均分至少相差 1 的候选。报告锚点排序一致率和全配对 Spearman。
- reference_student.py 比较从冻结 VoxSim 教师蒸馏的普通统计池化和注意力统计池化。两种教师目标为 16 维主分量坐标回归，以及加上原特征间相似度拟合。训练时按开发时长平均分选检查点。
- source_filter_audit.py 对同一句自然语音用 WORLD 重合成，分别改变基频、基频轮廓幅度或声道谱包络频率轴。结果相对无修改重合成作比较，另报告重合成本身的误差。此诊断没有模仿质量或听感距离真值。

## 从训练特征重建地图

`rank_map_candidate.py --fit-only --variant own --output-dir <新目录>` 从三个本项目前端的开发特征库重训 16 维地图。训练人第 1–4 句用于拟合，开发人第 5–8 句用于选步。输出独立权重和选步记录。`--variant combined` 还读取 VoxSim 的特征。两者都可以用 `rank_map_cli.py --artifact <权重> --variant <版本> build` 建索引。

`wave_map_cli.py --variant robust_student --device cuda build` 以单条语音的 FP32 推理建立设备增强学生的独立索引。同一入口的 `query <音频>` 在这个版本上定位。需要 WavLM 的固定主干及本地训练检查点 `runs/robust_ssl_consistency_177_1800/best.pt`。命令均在 vm_perceptual 目录调用 Python。窗口内补零会改变部分语音特征，单条推理能消除查询结果对同批其他音频的依赖。

## 新文本数量和录音设备

`nested_ssl_trial.py --repeat R` 在第 R 组五折分区上，分别用每人 2 句和 6 句训练相同的学生。人和句均分离，教师每折重新训练。`channel_folds.py --repeat R` 用每人 6 句和同样的内外层人员划分，比较无通道增强与通道一致性训练，两种方法均以干净开发句的相关性选步。两个试验依次运行 R=0 至 4、R=0 至 2，各对应 100 和 60 次训练；完整后运行同名的 summary 汇总代码（文本数实验为 `nested_ssl_summary.py`）。设备试验对同一批 4 名 VCTK 说话者、24 对实际双麦录音逐条测试，外层 JVS 语句也逐条计算。多折结果描述重复划分下的变化范围。

`physical_voice_quality.py` 在相同的 80 人开发语料和 25 次分折上检验显式声学指标。第一个候选只用 Praat 的平滑倒谱峰突出度（CPPS），第二个再加入 HNR、对数基频和有声帧 F3 中位数。量纲在训练人员内按四分位距缩放；五个融合权重包含零，由内层开发人选择。特征定义受跨语言感知实验启发，各论文的 CPP/CPPS 提取设置可能不同。[跨语言感知论文](https://pubs.asha.org/doi/10.1044/2026_JSLHR-25-01029) · [Praat 的 CPPS 定义](https://www.fon.hum.uva.nl/praat/manual/PowerCepstrogram__Get_CPPS___.html)
