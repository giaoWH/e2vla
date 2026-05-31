# 自定义 ALOHA 数据微调说明

这份文档记录如何用自定义 HDF5 数据重新微调 E2VLA。当前数据路径假设为：

```text
/nas_data_new/zzj/data_ssd/self_aloha/data_hdf5/pick_place_1031
```

预训练 checkpoint 假设为：

```text
/home/wh/e2vla/checkpoints/0927_e2vla_base_pretrain/ckpt_0600000.pt
```

## 1. 进入环境和项目目录

每次新开终端都先执行：

```bash
conda activate e2vla
cd /home/wh/e2vla
```

## 2. 先检查 HDF5 数据结构

训练代码不能直接吃任意 HDF5。它要求 HDF5 里有末端位姿、夹爪、时间戳、相机图像、相机内参和相机外参。

先运行下面的脚本查看第一条数据结构：

```bash
python - <<'PY'
import glob
import h5py

root = "/nas_data_new/zzj/data_ssd/self_aloha/data_hdf5/pick_place_1031"
files = glob.glob(root + "/**/*.h5", recursive=True)
files += glob.glob(root + "/**/*.hdf5", recursive=True)
files.sort()

print("num files:", len(files))
assert len(files) > 0, "No h5/hdf5 files found."
print("first file:", files[0])

with h5py.File(files[0], "r") as f:
    print("\nattrs:")
    for k, v in f.attrs.items():
        print(" ", k, "=", v)

    print("\nitems:")
    def visit(name, obj):
        if hasattr(obj, "shape"):
            print(" ", name, obj.shape, obj.dtype)
        else:
            print(" ", name, type(obj))
    f.visititems(visit)
PY
```

理想情况下，每个文件至少应包含类似字段：

```text
ee_pose              # (T, 4, 4) 或 (T, Nee, 4, 4)
gripper              # (T,) 或 (T, Nee)
timestamp            # (T,)

head_cam/rgb
head_cam/pose
head_cam/K

rh_cam/rgb
rh_cam/pose
rh_cam/K
```

并且 HDF5 attrs 里最好有：

```text
prompt_text
```

如果没有 `prompt_text`，代码会退化成默认 prompt：

```text
Do any possible actions
```

如果数据里只有 `joint_pos`，没有 `ee_pose`，不能直接微调当前模型。当前 E2VLA 的监督目标是未来末端执行器位姿轨迹，不是关节角轨迹。需要先通过机器人 FK 把关节角转换成 `ee_pose`。

## 3. 注册自定义 Dataset

编辑：

```text
/home/wh/e2vla/data_utils/datasets.py
```

在文件中加入一个新的数据集类。可以放在 `OpenOven` 类后面、`get_subclasses` 函数前面：

```python
class SelfAlohaPickPlace1031(H5DatasetMapBase):
    config = DataConfig(
        record_dt=1.0 / 10,
        sample_dt=1.0 / 10,
        output_image_hw=(224, 224),
        ee_indices=(0,),
        camera_names=("head_cam", "rh_cam"),
        sample_state_gaps=1,
        sample_camera_gaps=1,
        shuffle_cameras=False,
    )

    @classmethod
    def inst(cls):
        h5_files = glob.glob(
            "/nas_data_new/zzj/data_ssd/self_aloha/data_hdf5/pick_place_1031/**/*.h5",
            recursive=True,
        )
        h5_files += glob.glob(
            "/nas_data_new/zzj/data_ssd/self_aloha/data_hdf5/pick_place_1031/**/*.hdf5",
            recursive=True,
        )
        h5_files.sort()
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        return cls(h5_files)
```

这里的关键配置：

- `camera_names=("head_cam", "rh_cam")`：必须和 HDF5 里的相机 group 名一致。
- `output_image_hw=(224, 224)`：图像会 resize/crop 到 224x224。
- `record_dt=1.0/10`、`sample_dt=1.0/10`：按 10 Hz 采样。
- `ee_indices=(0,)`：只训练第 0 个末端执行器。
- `sample_state_gaps=1`：未来动作间隔为 `sample_dt * sample_state_gaps`。

如果你的 HDF5 相机名不是 `head_cam` / `rh_cam`，这里要改成实际名字。

## 4. 添加训练配置

编辑：

```text
/home/wh/e2vla/configs.py
```

在已有 `CONFIGS[...]` 后面加入：

```python
CONFIGS["finetune_self_aloha_pick_place_1031"] = TrainConfig(
    dataset_classes=[datasets.SelfAlohaPickPlace1031],
    dataset_weights=[1],
    sample_multiplex=1000,
    num_warmup=int(2e3),
    save_interval=int(10e3),
    max_iterations=int(70e3),
)
```

如果数据量比较大，可以把 `sample_multiplex` 改小，例如：

```python
sample_multiplex=1
```

如果只是先验证流程，可以把 `max_iterations` 临时改小，例如：

```python
max_iterations=int(1000)
```

## 5. 先测试 Dataset 是否能读

也可以先运行依赖检查脚本，确认训练依赖、CUDA、checkpoint、数据目录和训练配置是否齐全：

```bash
python scripts/check_train_deps.py
```

如果想顺便实际用 `torch.load` 在 CPU 上检查 checkpoint，可以加：

```bash
python scripts/check_train_deps.py --load-ckpt
```

启动正式训练前，先跑一个最小读取测试：

```bash
python - <<'PY'
from data_utils.datasets import SelfAlohaPickPlace1031

d = SelfAlohaPickPlace1031.inst()
print("dataset length:", len(d))

x = d[0]
for k, v in x.items():
    if hasattr(v, "shape"):
        print(k, v.shape, v.dtype)
    else:
        print(k, type(v), v)
PY
```

如果这一步报错，先修数据格式或 Dataset 配置，不要直接开始训练。

## 6. 启动微调

用预训练 checkpoint 微调。这里 batch size 固定使用 `32`：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_self_aloha_pick_place_1031 \
  --pretrained_ckpt /home/wh/e2vla/checkpoints/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  -s finetune_pick_place_1031 \
  --bs 32
```

当前 `train.py` 默认只使用 `cuda:0`，不会自动使用 4 张 4090。可以通过 `CUDA_VISIBLE_DEVICES=1`、`2`、`3` 换卡，但 batch size 仍保持 `32`。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config finetune_self_aloha_pick_place_1031 \
  --pretrained_ckpt /home/wh/e2vla/checkpoints/0927_e2vla_base_pretrain/ckpt_0600000.pt \
  -s finetune_pick_place_1031 \
  --bs 32 \
  --workers 4
```

## 7. 输出位置

训练日志会保存到：

```text
/home/wh/e2vla/logs/E2VLA/finetune_pick_place_1031
```

checkpoint 会保存到：

```text
/home/wh/e2vla/checkpoints/E2VLA/finetune_pick_place_1031
```

目录里会有：

```text
ckpt_latest.pt
ckpt_best.pt
ckpt_0010000.pt
...
YYYYMMDDHHMM.json
```

后续启动 `remote_service` 时，应该使用微调后的 checkpoint，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt /home/wh/e2vla/checkpoints/E2VLA/finetune_pick_place_1031/ckpt_latest.pt \
  --uri e2vla \
  --ns_host 10.15.194.83 \
  --ns_port 9090 \
  --host 10.15.194.83 \
  --port 9091
```

## 8. 常见问题

### 找不到数据

如果报：

```text
AssertionError
```

并且前面打印：

```text
num samples of SelfAlohaPickPlace1031: 0
```

说明 glob 路径不对，或者文件后缀不是 `.h5` / `.hdf5`。

### 缺少 prompt_text

如果 HDF5 attrs 里没有 `prompt_text`，训练仍能跑，但语言输入会变成：

```text
Do any possible actions
```

这对语言条件控制不友好。最好在 HDF5 attrs 里写入真实任务指令，例如：

```text
pick up the pepper and place it on the plate
```

### 缺少 ee_pose

当前模型训练目标是未来末端位姿。没有 `ee_pose` 就不能直接用这套训练代码，需要先用机器人运动学从 `joint_pos` 计算末端位姿。

### 相机名不匹配

如果报类似：

```text
KeyError: 'head_cam'
```

说明 `camera_names` 配置和 HDF5 里的 group 名不一致。检查 HDF5 结构后修改 `SelfAlohaPickPlace1031.config.camera_names`。

### 显存不足

batch size 需要固定为 `32`，不要通过降低 `--bs` 解决。可以先换更空闲的卡：

```bash
CUDA_VISIBLE_DEVICES=1
```

如果单卡仍然 OOM，需要再考虑训练代码是否要改成梯度累积或多卡训练；当前仓库的 `train.py` 默认只用一张卡。
