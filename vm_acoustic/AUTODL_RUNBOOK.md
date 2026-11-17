# AutoDL运行手册：物理测量试验阶段

代码已在Windows/Python3.12/CPU执行；Linux/CUDA和AutoDL实例尚未实际部署。当前可迁移的是测量验证与物理预测头，不是训练完成的音色/语气地图。

## 先验证环境

建议Python3.12环境。在AutoDL选择适配驱动的PyTorch镜像，参考[官方依赖说明](https://www.autodl.com/docs/deps/)及[PyTorch版本安装表](https://pytorch.org/get-started/previous-versions/)。本地工程检查使用torch2.9.1+cpu；云端GPU构建需要依据实例驱动安装兼容CUDA版本，不复制Windows `.deps`。

在仓库根目录执行：

```bash
python -m pip install -r vm_acoustic/requirements.txt
# CPU复现实验；GPU实例用官方对应CUDA命令代替此行
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python vm_acoustic/voice_space.py
python vm_acoustic/train_physics.py --smoke --device cpu --output vm_acoustic/training_runs/smoke
```

不要把CPU轮子的安装行直接用于希望启用CUDA的环境。`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`检查实际运行后端。

## 数据和复现

包内不包含第三方原始音频和本地DLL。下载器从记录的来源重建数据，Hillenbrand固定commit和SHA256；VCTK下载有100MB上限及ZIP CRC检查。

```bash
python vm_acoustic/fetch_reference.py
python vm_acoustic/reference_benchmark.py
python vm_acoustic/prepare_physics_training.py
python vm_acoustic/fetch_vctk_validation.py
python vm_acoustic/build_sentence_validation.py
python vm_acoustic/benchmark_tracker.py
python vm_acoustic/validate_tracker_ood.py
python vm_acoustic/evaluate_tracker_pairs.py
python vm_acoustic/fetch_vctk_validation.py --confirmation
python vm_acoustic/confirm_tracker.py
```

公开数据访问依赖实例网络，可在本地完成下载再用[SCP](https://www.autodl.com/docs/scp/)传输。使用前保留源数据版权、引用和许可文件；不要把原始语音自动提交到GitHub。

## 有目的的小训练

准备出的物理预测数据为318训练token/53人、66开发token/11人、78测试token/13人，完全排除本轮96条参考验证的16个人。数据量只够小型测量实验，远不足以代表通用音色空间。

```bash
python vm_acoustic/train_physics.py --train \
  --manifest vm_acoustic/data_reference/training_manifest.json \
  --device cuda --epochs 100 \
  --output /root/autodl-tmp/vocalmap_runs/physics_v0
```

本轮已在本地CPU完成一次真实数据100epoch试验：测试集共振峰中位相对误差4.66%，劣于同目标Burg基线1.16%，因此不推荐直接扩大这个网络。上述CUDA部署尚未实测，正式音色/语气模型尚未训练。验证集选择best.pt；test分组不进入训练/早停。训练脚本当前为全批小数据实现，扩容后需加入按人采样、批处理、明确的学习率调度和性能测试。

```bash
python vm_acoustic/train_physics.py --train \
  --manifest vm_acoustic/data_reference/training_manifest.json \
  --device cuda --epochs 50 \
  --resume /root/autodl-tmp/vocalmap_runs/physics_v0/last.pt \
  --output /root/autodl-tmp/vocalmap_runs/physics_v0
```

恢复时校验数据哈希，并恢复模型、优化器和epoch；当前网络无随机增强/dropout，固定种子。增加随机训练组件后还需保存全部随机数状态。

最终评估（只在开发选择完成后执行，不以测试结果继续调参）：

```bash
python vm_acoustic/evaluate_physics.py \
  --manifest vm_acoustic/data_reference/training_manifest.json \
  --checkpoint /root/autodl-tmp/vocalmap_runs/physics_v0/best.pt \
  --output /root/autodl-tmp/vocalmap_runs/physics_v0/evaluation
```

`--epochs`表示本次新增轮数；恢复必须写入原checkpoint目录，保留best.pt及累计历史。全新训练必须使用空输出目录，以免静默覆盖。

模型约74,115参数，CPU已经能跑；为这个阶段专门租大显存/多卡没有必要。未来冻结大编码器可先缓存特征，小头训练单GPU足够作为试跑方案；具体GPU显存由片长、batch和编码器决定，需要实测。

## 保存和退出

训练输出包括best.pt、last.pt、run.json；保存数据清单、脚本版本与评估报告。在tmux会话中运行可保持断开SSH后的任务；训练输出放数据盘并下载备份，见[AutoDL数据说明](https://www.autodl.com/docs/instance_data/)。本轮没有开实例、付款、上传录音或发布GitHub仓库。

正式音色与语气训练的目标和关卡见 `MODEL_CONTRACT.md`。以该文件的研究就绪度为准，不能把“脚本能运行”当作模型目标已成立。
