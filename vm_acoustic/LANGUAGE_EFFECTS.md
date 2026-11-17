# 中文、日文、英文对音色测量的影响

2026-11-17研究备忘。当前实录只有VCTK英语，**没有实测中日英跨语言稳定性**。以下将文献证据与项目设计分开。

## 为什么语言会影响

傅里叶变换、共振峰和声源分析的数学定义不依赖语言，但人说不同语言时的发音、音素分布、音高和发声方式会变。因此同一个人的整句F1/F2、F0、CPP/HNR统计会变；这不全是设备噪声，也不代表换了一个声带或身份。

1. 音素组成和习惯发音不同。相同拉丁字母或近似IPA类别，不保证声学目标完全相同，英语内部口音也有差异。不能直接将所有语言拼写i/u的音合并为英文IY/UW，也不能用当前美式英语词典对齐中文/日文。
2. 普通话的词汇声调会与发声状态和元音交互。McGregor-Smith与Yanushevskaya（Speech Prosody 2026）的12位普通话说话者实验考察6个单元音与4个声调，报告HNR、CPPS、F0等存在元音及声调条件差异。是初步样本，不能当成所有汉语方言规律；CPPS也不等同本项目每帧CPP实现。本文本轮读到官方摘要，未核完全文。
3. 日语高元音在某些环境下会清化。Tanner、Sonderegger与Torreira（2019）的东京日语研究使用自然语料考察相关时长和清化现象。若某段/i,u/不周期振动，当前要求有效F0/HNR的六参数联合估计会拒绝它；应该记录“无声/缺失”，不能强行填成正常有声元音，或把清化误判为说话者音色改变。清化并非每个高元音都发生，还受语境与说话方式影响。
4. 同一说话者跨语言也可能改变发音设置。Lo（Interspeech 2021）分析60位男性英法双语者，发现长期共振峰分布存在语言系统差异，同时保留明显的说话者一致性；跨语言比较的区分效果低于同语言。该证据来自英法，不能直接当作中日英已被验证。

## 对我们数学模型的影响（设计假设）

将测量写为：q = f(z_voice, vowel/articulation, language, tone/accent, speaking_state) + g(channel) + error。这是建模框架，不主张各因素真实可线性相加或完全可分。源/滤波器和声调可能有交互。

距离应该条件化：先比较相匹配且可靠的发音条件，再把共同支持的条件以固定规则合并。每个结果携带元音覆盖、语言、声调/重音、设备和不确定度。没有共同条件时返回证据不足，避免给一个看似精确却混入语言差异的距离。

可进一步拟合层级模型 q_svlt = μ_vl + B_v z_s + tone/state terms + channel terms + interactions + ε。必须有“同人多语言”和跨人的交叉数据才能区分语言效应与说话者效应；只有不同国家各一批人会混淆这些因素。

不能通过每个人都把自己的F1/F2均值归零来消除语言，因为这也会消掉我们想保留的声道尺度/音色信息。先保留原始物理参数与条件化参数两份，检查哪些变化应该被保留。

AI可采用共享声学编码器、语言/音素/声调条件和独立质量分支；以后才考虑语言对抗等去混淆方法。身份保持不等于音色完全不变，不能仅凭“同一个人”把所有语言和声线强行压成同一点。

## 最小多语言验证建议（未执行）

- 同一批会说两种或三种语言的人，以相同设备、录音日、距离、增益录制；每语言再重复一次。优先控制口音和熟练度，而非按国籍推断语言。
- 每人录自然单元音/载句、自然短句和重复持元音三类。持元音用于物理测量校验，自然短句用于检验迁移，不能用前者代替后者。
- 为各语言用相应词典/音素模型对齐，人工核验少量边界；共同发音条件由语音学审查决定。中文按声调分层；日文显式标记清化；英语按口音/重音及单/双元音区分。
- 按人/句/录音日留出，比较：同人同语言重复误差、同人跨语言漂移、同条件异人差异、可靠覆盖率。只有跨语言漂移接近同语言重复误差、同时保留可核实的声学差异，才有证据支持通用地图。

第一阶段可先做一个有交叉条件的小实验；不能用目前8位英语者证明语言不影响。听众相似度仍可用于后续感知验证，独立于物理参数和边界校验。

## 来源与本轮阅读范围

- [Mandarin vowels: Intrinsic properties and interaction with tone, 2026](https://www.isca-archive.org/speechprosody_2026/mcgregorsmith26_speechprosody.html)：官方摘要。
- [Durational Evidence That Tokyo Japanese Vowel Devoicing Is Not Gradient Reduction, 2019](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.00821/full)：研究正文页面，阅读相关语料、清化与结论部分，非全部文献系统综述。
- [Cross-Linguistic Speaker Individuality of Long-Term Formant Distributions, 2021](https://www.isca-archive.org/interspeech_2021/lo21_interspeech.html)：官方摘要。
- [Disentangled Representation Learning for Multilingual Speaker Recognition, 2023](https://www.isca-archive.org/interspeech_2023/nam23_interspeech.html)：官方摘要，提出双语评测与语言去混淆表征；身份任务的成果不等同音色地图有效性。
