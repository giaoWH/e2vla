# remote_service 启动说明

这份项目里的 `infer_utils/remote_service.py` 用来把训练好的 E2VLA checkpoint 启动成一个 Pyro4 RPC 推理服务。客户端通过 `uri` 连接服务，发送观测帧，然后获取未来动作轨迹。

## 1. 启动前检查

下面所有命令都建议在服务器上执行，并且两个终端都先进入同一个 conda 环境和项目根目录。假设 conda 环境名是 `e2vla`，项目路径是 `/home/wh/e2vla`：

```bash
conda activate e2vla
cd /home/wh/e2vla
```

其中第二个终端启动 `remote_service` 时必须在项目根目录 `/home/wh/e2vla` 下运行，因为命令使用的是：

```bash
python -m infer_utils.remote_service
```

如果不在项目根目录，Python 可能找不到 `infer_utils`、`models`、`data_utils` 等本项目模块。

确认 Python 环境已经装好项目依赖，并且可以 import PyTorch、transformers、diffusers、Pyro4、opencv 等包。

确认你有 E2VLA checkpoint。`remote_service.py` 不会自动下载策略模型权重，必须手动指定：

你现在已经准备好的 checkpoint 和配置文件是：

```text
/home/wh/e2vla/checkpoints/E2VLA/finetune_pick_place_1031/
      ckpt_best.pt
      202605311746.json
```

注意：checkpoint 所在目录里还需要有训练时保存的 `.json` 配置文件，否则服务启动时会报：

```text
No config files found in ...
```

视觉 backbone 权重，例如 DINOv2 和 SigLIP，如果本地没有缓存，代码会尝试通过 Hugging Face 或 `HF_ENDPOINT=https://hf-mirror.com` 自动下载。但 E2VLA 自己的 `.pt` checkpoint 不会自动下载。

## 2. 不要直接按文件路径运行

不推荐这样运行：

```bash
python infer_utils/remote_service.py
```

因为 `remote_service.py` 里有相对导入：

```python
from .planner import TrajPlanner
```

直接按文件路径运行容易报：

```text
ImportError: attempted relative import with no known parent package
```

正确方式是从项目根目录用模块方式运行：

```bash
python -m infer_utils.remote_service ...
```

## 3. 本机启动方式

如果客户端和服务端都在同一台机器上，先开一个终端启动 Pyro4 naming server：

终端 1：

```bash
conda activate e2vla
cd /home/wh/e2vla
pyro4-ns
```

默认监听：

```text
localhost:9090
```

再开第二个终端启动模型服务：

终端 2：

```bash
conda activate e2vla
cd /home/wh/e2vla
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt /home/wh/e2vla/checkpoints/E2VLA/finetune_pick_place_1031/ckpt_best.pt \
  --uri e2vla \
  --ns_host localhost \
  --ns_port 9090 \
  --host localhost
```

参数说明：

- `--ckpt`：必填，E2VLA checkpoint 路径。
- `--uri`：服务注册名，客户端要用同一个名字连接。
- `--ns_host` / `--ns_port`：Pyro4 naming server 地址和端口。
- `--host` / `--port`：模型服务本身监听的地址和端口。`--port 0` 表示自动分配端口。
- `--ema`：如果 checkpoint 里有 EMA 权重并且想用 EMA 推理，可以加这个参数。
- `--ensemble`：轨迹平滑用的 ensemble 历史长度，默认是 `4`。

## 4. 局域网/服务器启动方式

如果客户端在另一台机器上，需要让 naming server 和模型服务监听服务器 IP。

假设服务器 IP 是：

```text
10.15.194.83
```

终端 1：

```bash
conda activate e2vla
cd /home/wh/e2vla
pyro4-ns -n 0.0.0.0 -p 9091
```

终端 2：

```bash
conda activate e2vla
cd /home/wh/e2vla
CUDA_VISIBLE_DEVICES=0 python -m infer_utils.remote_service \
  --ckpt /home/wh/e2vla/checkpoints/E2VLA/finetune_pick_place_1031/ckpt_best.pt \
  --uri e2vla \
  --ns_host 10.15.194.83 \
  --ns_port 9091 \
  --host 10.15.194.83 \
  --port 40911
```

客户端连接时也要使用同样的 `uri`、`ns_host`、`ns_port`：

```python
from shm_transport import get_shm_proxy

policy = get_shm_proxy(
    uri_name="e2vla",
    ns_host="10.15.194.83",
    ns_port=9091,
)
```

## 5. 快速验证

服务启动成功后，日志里应该能看到类似信息：

```text
[INFO] Use config file ...
[INFO] model = ...
[INFO] Load weights from iter: ...
[INFO] Start service, uri = PYRO:...
```

如果看到 `Start service`，说明服务已经注册到 naming server。

## 6. 常见问题

### 没传 checkpoint

报错：

```text
Please specify a valid ckpt path.
```

解决：启动时加上 `--ckpt /path/to/ckpt.pt`。

### checkpoint 目录没有 json

报错：

```text
No config files found in ...
```

解决：把训练时保存的 `.json` 配置文件放到 checkpoint 同目录。

### 直接运行文件导致相对导入失败

报错：

```text
ImportError: attempted relative import with no known parent package
```

解决：用模块方式运行：

```bash
python -m infer_utils.remote_service
```

### 客户端连不上服务

检查：

- `pyro4-ns` 是否还在运行。
- `--uri` 是否和客户端一致。
- `--ns_host` / `--ns_port` 是否一致。
- 服务器防火墙是否放行端口。
- 如果跨机器访问，`--host` 不要写 `localhost`。

### 真实机器人客户端调用 `get_action_torch`

当前这份 `remote_service.py` 暴露的是：

```python
get_action(...)
```

不是：

```python
get_action_torch(...)
```

如果真实机器人侧代码调用 `get_action_torch`，说明服务器上可能需要另一版服务代码，或者需要额外写一层适配：把 `get_action()` 输出的末端位姿轨迹转换成关节轨迹。
