
# 一、环境配置


由于jetson orin nano super是arm架构，很多组件需要自己编译，而且会有比较多的不同组件之间的依赖问题，所以最方便的还是直接使用docker。jetson orin nano super 的AI docker环境配置有两个难点：

（1）CPU是arm架构
（2）super要使用jetpack 6.2[L4T 36.4.3]版本，因为版本太新，这个版本很多开源项目还没有发布对应的docker image，大部分的docker image都是基于jetpack 5.x构建的。

下面我们来介绍部署yolov10需要的pytorch、tensorrt等组件的一个docker image。

## 系统设置

check一下在jetosn orin nano super上安装的是JetPack 6。可以通过`jtop`工具来查看自己安装的jetpack版本。

![](https://borninfreedom.github.io/images/2025/03/jetson/1.png)


## 下载jetson-containers工具

jetson-containers可以通过模块化的方式来自动构建image，但是jetson-containers也有构建好的包含所有我们使用组件的image，我们用的就是他们构建好的image。

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```
安装脚本会提示您输入sudo密码，并会安装一些Python依赖项，并通过在`/usr/local/bin`下建立链接的方式将诸如autotag之类的工具添加到`$PATH`中（如果您移动了jetson-containers存储库，请再次运行此步骤）。


## 修改Docker默认运行时为nvidia

这一步建议做，不然每次启动container时，都要加上--runtime=nvidia，例如下面的启动指令，就要加上--runtime。

```bash
sudo docker run --runtime nvidia --gpus all --net host --ipc host -it --name pytorch_ngc_v2 -v /home:/home nvcr.io/nvidia/pytorch:25.01-py3-igpu
```

修改`/etc/docker/daemon.json`文件，将`"default-runtime": "nvidia"`添加到`/etc/docker/daemon.json`配置文件中：
``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

然后重启Docker服务：
```bash
$ sudo systemctl restart docker
```
可以通过查看`docker info`来确认更改：

```bash
$ sudo docker info | grep 'Default Runtime'
Default Runtime: nvidia
```

## 重新定位Docker数据根目录

这一步如果jetson设备已经额外安装了硬盘，就一般不需要做了。或者自己的docker安装位置分区足够，也不需要做。


容器可能会占用大量磁盘空间。如果有可用的外部存储，建议将Docker容器缓存重新定位到更大的驱动器上（如果可能的话，NVME是首选）。如果尚未格式化，请将驱动器格式化为ext4格式，并使其在启动时挂载（即应在`/etc/fstab`中）。如果在Docker守护进程启动之前，驱动器在启动时未自动挂载，那么Docker将无法使用该目录。

将现有的Docker缓存从`/var/lib/docker`复制到您选择的驱动器上的目录（在本例中为`/mnt/docker`）：
```bash
$ sudo cp -r /var/lib/docker /mnt/docker
```
然后在`/etc/docker/daemon.json`中添加您的目录作为`"data-root"`：
``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia",
    "data-root": "/mnt/docker"
}
```
然后重启Docker服务：

```bash
$ sudo systemctl restart docker
```

可以通过查看`docker info`来确认更改：

```bash
$ sudo docker info | grep 'Docker Root Dir'
Docker Root Dir: /mnt/docker
...
Default Runtime: nvidia
```

## docker pull设置代理


```python
sudo vi /etc/systemd/system/docker.service.d/http-proxy.conf
```

在文件中添加：（把其中的IP和端口换成自己的代理IP和端口）
```bash
[Service]
Environment="HTTP_PROXY=http://192.168.1.10:7890"
Environment="HTTPS_PROXY=http://192.168.1.10:7890"
```

然后
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 增大swap分区

这一步建议做。因为jetson orin nano super只有8G的显存，对于跑更大的模型，如果swap分区足够大，也是可以跑得开的，只是慢一点罢了。

如果您要构建容器或处理大型模型，建议挂载交换分区（通常与开发板上的内存量相关）。运行以下命令来禁用ZRAM并创建一个交换文件：
``` bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
```
> 如果有可用的NVME存储，最好在NVME上分配交换文件。

然后在`/etc/fstab`的末尾添加以下行，以使更改永久生效：
``` bash
/mnt/16GB.swap  none  swap  sw 0  0
```

## 禁用桌面图形用户界面

如果内存不足，您可能需要尝试禁用Ubuntu桌面图形用户界面（GUI）。这将释放窗口管理器和桌面所占用的额外内存（对于Unity/GNOME约为800MB，对于LXDE约为250MB）。

在我的机器上，图形用户界面占用了450M左右的memory。将它关掉还是能省很多的memory的。一般我都是不用图形化的时候就先关掉，用的时候再打开。
![](https://borninfreedom.github.io/images/2025/03/jetson/2.png)



可以临时禁用桌面，在控制台中运行命令，然后在需要时重新启动桌面：
``` bash
$ sudo init 3     # 停止桌面
# 使用Ctrl+Alt+F1、F2等组合键让用户重新登录到控制台
$ sudo init 5     # 重新启动桌面
```
如果希望在重启后该设置仍然生效，可以使用以下命令来更改启动行为：
``` bash
$ sudo systemctl set-default multi-user.target     # 启动时禁用桌面
$ sudo systemctl set-default graphical.target      # 启动时启用桌面
```

## 将用户添加到Docker组

由于Ubuntu用户默认不在`docker`组中，他们需要使用`sudo`来运行docker命令（构建工具在需要时会自动执行此操作）。因此，在构建过程中可能会定期要求您输入sudo密码。

相反，您可以按如下方式将用户添加到docker组：
```bash
sudo usermod -aG docker $USER
```
然后关闭/重新启动终端（或注销），您应该能够在无需使用sudo的情况下运行docker命令（例如`docker info`）。

## 设置电源模式

根据Jetson设备可用的电源来源（即墙上电源或电池），您可能希望将Jetson设置为最大功率模式（MAX-N），以获得Jetson设备的最高性能。您可以使用[`nvpmodel`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls)命令行工具来实现，或者通过Ubuntu桌面使用[nvpmodel图形用户界面小部件](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#nvpmodel-gui)（或者使用jetson-stats中的[`jtop`](https://github.com/rbonghi/jetson_stats)）。
```bash
# 检查当前电源模式
$ sudo nvpmodel -q
NV Power Mode: MODE_30W
2

# 将其设置为模式0（通常是最高模式）
$ sudo nvpmodel -m 0

# 如有必要，重启并确认更改
$ sudo nvpmodel -q
NV Power Mode: MAXN
0
```
![](https://borninfreedom.github.io/images/2025/03/jetson/3.png)
我的当前电源模式是功率最高的设置，从jtop的右下角可以看到实时的功率信息。

有关不同Jetson设备可用的电源模式表，以及[`nvpmodel`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#power-mode-controls)工具的文档，请参阅[此处](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency)。 

## 拉取docker镜像

我们使用jetson-containers工具来自动匹配我们的机器，这个命令会查看当前jetson的jetpack版本以及当前host的其他组件的版本，来自动选择合适的docker image。

```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag nanoowl)
```
这个命令在我的机器上，其实是直接拉取的`dustynv/nanoowl:r36.4.0`这个镜像。


拉取完成后，镜像会自动运行，我们可以直接ctrl+D退出，使用我们的自定义命令重新打开。

```bash
sudo docker run --runtime nvidia --gpus all --net host --ipc host -it --name ai_all_in_one  -v /home:/home dustynv/nanoow:r36.4.0
```

到这里已经配置完成了我们部署yolov10基本的环境了。

# 二、配置docker内使用usb摄像头


## 1. 在host上接入摄像头

第一件任务就是要判断摄像头的种类与数量，用最简单的 “ls /dev/video*” 指令并不能分辨其种类，因此最好的方法还是使用 v4l2 工具。请先执行以下指令安装这个工具：

```bash
sudo  apt   install  -y  v4l-utils
```

安装好之后，请执行以下指令：

```bash
v4l2-ctl  --list-devices
```

如果检测到以下 “imx219” 之类的信息，表示这个摄像头为 CSI 类型：
![](https://borninfreedom.github.io/images/2025/04/usb/1.png)

如果检测到以下 “USB Camera” 信息的，就表示为 USB 摄像头：
![](https://borninfreedom.github.io/images/2025/04/usb/2.png)

在 Jetson Orin 开发套件的 USB 摄像头都会占用 2 个 video 口，例如上图中的一台 USB 摄像头占用 video0 与 video1 两个端口，但实际能调用作为输入功能的是第一个 video0 的编号，如果设备上有多个摄像头的时候，就需要特别注意这些细节。

## 2. 相关配置

### 2.1 确认 DISPLAY 环境变量（关键步骤）​​

在host上执行

```bash
export DISPLAY=:0  # 默认本地显示器（常见于物理机直接操作）
```

### 2.2 配置 X Server 访问权限​​

安装 X11 基础工具（宿主机和容器内均需要）：

```bash
sudo apt install x11-xserver-utils xauth x11-apps libgl1-mesa-glx libgtk-3-0
```

```bash
# 在宿主机（非容器内）执行以下命令
xhost +local:docker  # 允许所有本地 Docker 容器访问 X 服务
```

检查 X 服务是否运行​​：

```bash
ps aux | grep Xorg  # 确认 Xorg 进程存在
```

## 3. docker run指令

确保 Docker 命令包含必要的 X11 参数：

```bash
docker run -it \
  --env DISPLAY=$DISPLAY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0  \  # 若使用摄像头
  your_image
```

完整的指令如下：

```bash
sudo docker run --runtime nvidia --gpus all --net host --ipc host -it \
--device /dev/video0 --device /dev/video1 --name yolo_usb3 \
-v /home:/home -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
dustynv/nanoowl:r36.4.0
```

进入到docker container后，记得安装 X11基础工具：

```bash
sudo apt install x11-xserver-utils xauth x11-apps libgl1-mesa-glx libgtk-3-0
```

# 4. docker container内测试

在container中，执行
```bash
xclock  # 若弹出时钟窗口，则 X11 转发成功
```
此时应该会弹出一个时钟的窗口。若弹出，代表X11服务从host到docker转发成功。

如果使用opencv打开usb摄像头，一定记得使用`cap = cv2.VideoCapture(0, cv2.CAP_V4L2)`来打开摄像头。

# 三、yolov10代码修改以使用tensorrt

你提供的代码，其中使用tensorrt导出tensorrt模型的代码比较老了，是适配tensorrt 7的，我们在docker内使用的是tensorrt 10，因此要对其进行更改。

首先编写导出tensorrt模型的代码：这里直接导出fp16的模型，精度和fp32差不多，但是性能可以提升1倍。
```python
from ultralytics import YOLOv10
model = YOLOv10("runs/V10train/exp3_fp16/weights/best.pt")
model.export(format="engine",half=True) # half=True，导出fp16的模型，性能会比fp32的有1倍的加速
# 然后重新把tensorrt的模型load进来
model = YOLOv10("runs/V10train/exp3_fp16/weights/best.engine")
model.predict(source="test_img", imgsz=640, conf=0.05,save=True)   #整个文件夹测试
```

然后需要修改ultralytics源码内部对于tensorrt导出的代码，以让其支持tensorrt 10.

代码路径是`ultralytics/engine/exporteer.py`，搜索`def export_engine`。

export_engine的原始实现是：

```python
  @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
            import tensorrt as trt  # noqa

        check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0

        self.args.simplify = True

        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = self.args.workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
            config.add_optimization_profile(profile)

        LOGGER.info(
            f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and self.args.half else 32} engine as {f}"
        )
        if builder.platform_has_fast_fp16 and self.args.half:
            config.set_flag(trt.BuilderFlag.FP16)

        del self.model
        torch.cuda.empty_cache()

        # Write file
        with builder.build_engine(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine.serialize())

        return f, None
```

这里面的大部分API都是兼容tensorrt 7的比较旧的，需要更新其为新的API，但是代码也要同时兼容老的API。新的代码修改为：

```python
    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """YOLO TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,!=10.1.0")
            import tensorrt as trt  # noqa
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        # Engine builder
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int((self.args.workspace or 0) * (1 << 30))
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8

        # Optionally switch to DLA if enabled
        if dla is not None:
            # if not IS_JETSON:
            #     raise ValueError("DLA is only available on NVIDIA Jetson devices")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Read ONNX file
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        # Network inputs
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # minimum input shape
            max_shape = (*shape[:2], *(int(max(1, self.args.workspace or 1) * d) for d in shape[2:]))  # max input shape
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset,  # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = dataset
                    self.data_iter = iter(dataset)
                    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    self.batch = batch
                    self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    """Get the calibration algorithm to use."""
                    return self.algo

                def get_batch_size(self) -> int:
                    """Get the batch size to use for calibration."""
                    return self.batch or 1

                def get_batch(self, names) -> list:
                    """Get the next batch to use for calibration, as a list of device memory pointers."""
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        # Return [] or None, signal to TensorRT there is no calibration data remaining
                        return None

                def read_calibration_cache(self) -> bytes:
                    """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()

                def write_calibration_cache(self, cache) -> None:
                    """Write calibration cache to disk."""
                    _ = self.cache.write_bytes(cache)

            # Load dataset w/ builder (for batching) and calibrate
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.args.batch,  # TensorRT INT8 calibration should use 2x batch size
                cache=str(self.file.with_suffix(".cache")),
            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Free CUDA memory
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # Write file
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())

        return f, None
```

这里主要的修改为：

(1)根据不同的tensorrt版本来自动确定分配内存池的API

```python
       if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
```

（2）在支持DLA的板子上使用DLA来部署模型

```python
        # Optionally switch to DLA if enabled
        if dla is not None:
            # if not IS_JETSON:
            #     raise ValueError("DLA is only available on NVIDIA Jetson devices")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
```

(3)根据不同的tensorrt版本，来确定build tensorrt engine的API

```python
 build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine if is_trt10 else engine.serialize())
```

导出tensorrt engine后，然后运行tensorrt的model的时候，同样需要对其推理的代码进行修改以适配tensorrt 10.

ultralytics推理时，会运行到`ultralytics/nn/autobacked.py`中。针对tensorrt backend，具体的代码是`elif engine`分支。原始的实现是：

```python
      TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
                import tensorrt as trt  # noqa
            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
```

我们对其修改，添加tensorrt 10的支持。修改后的代码为：

```python
        # TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")

            # if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.0"):
            #     # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 with Python <= 3.8.0
            #     check_requirements("numpy==1.23.5")

            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt  # noqa
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
                model = runtime.deserialize_cuda_engine(f.read())  # read engine

            # Model context
            try:
                context = model.create_execution_context()
            except Exception as e:  # model is None
                LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
```

其中主要的修改是修改适配tensorrt 10的API：


```python
 is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
```

```python
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
```

# 四、编写可视化界面

可视化界面使用gradio来搭建。

app的代码如下：

```python
import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import gradio as gr
import cv2
# from gradio_webrtc import WebRTC
from fastrtc import WebRTC
from twilio.rest import Client
import os
import numpy as np
import time
import PIL.Image as Image
from collections import deque
from dotenv import load_dotenv
load_dotenv()  # 添加在代码开头

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    print('USE twilio')
    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }
else:
    rtc_configuration = None

# FPS 计算类
class FPS:
    def __init__(self, avg_frames=30):
        self.timestamps = deque(maxlen=avg_frames)  # 用于存储时间戳
    
    def update(self):
        self.timestamps.append(time.time())  # 添加当前时间戳
        
    def get(self):
        if len(self.timestamps) < 2:
            return 0  # 如果时间戳不足，返回 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])  # 计算平均 FPS

fps_counter = FPS()  # 创建 FPS 计数器实例


from ultralytics import YOLOv10

# model = YOLOv10("runs/V10train/exp3/weights/best.pt")
model = YOLOv10("runs/V10train/exp3_fp16/weights/best.engine")
print(f'{model = }')


def yolov10_inference_video(video_path):
    fps_counter = FPS()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 更新 FPS 计数器
        fps_counter.update()

        # 模型推理
        results = model.predict(source=frame, imgsz=640, conf=0.05)
        r = results[0]
        im_bgr = r.plot()  # Ultralytics 默认输出 BGR numpy

        # 计算 FPS
        fps = fps_counter.get()
        cv2.putText(im_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将帧返回给 Gradio
        yield im_bgr  # 使用 yield 逐帧返回推理结果

    cap.release()


def yolov10_inference(image):
    fps_counter.update()
    if 1:
        results = model.predict(source=image, imgsz=640, conf=0.05,save=False)   #整个文件夹测试
    
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])

        # 计算 FPS
        fps = fps_counter.get()
        print(f"FPS: {fps:.1f}")  # 在控制台打印 FPS

        return im

# def yolov10_inference_webrtc(frame: np.ndarray) -> np.ndarray:
#     # frame: BGR numpy
#     results = model.predict(source=frame, imgsz=640, conf=0.05)
#     r = results[0]
#     im_bgr = r.plot()      # Ultralytics 默认输出 BGR numpy
#     return im_bgr          # Gradio 会自动把 BGR 转成 RGB 展示



def app():
    with gr.Blocks():
        with gr.Tabs():  # 添加选项卡
            with gr.TabItem("WebRTC Stream"):  # WebRTC 输入选项卡
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(sources=["webcam"],type="pil", streaming=True)
                    image.stream(
                        fn=yolov10_inference, inputs=[image], outputs=[image],stream_every=0.1
                    )
                    #     image = WebRTC(label="Stream", rtc_configuration=rtc_configuration, height=640, width=640)
                    # image.stream(
                    #     fn=yolov10_inference, inputs=[image], outputs=[image], time_limit=500
                    # )
            with gr.TabItem("Image Upload"):  # 图片输入选项卡
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        output_image = gr.Image(label="Output Image",type="pil")
                    input_image.change(
                        fn=yolov10_inference, inputs=[input_image], outputs=[output_image]
                    )
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_output = gr.Video(label="Output Video")
                    video_input.change(
                        fn=lambda video_path: list(yolov10_inference_video(video_path)),inputs=[video_input], outputs=[video_output]
                    )



gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(server_name = '0.0.0.0')
```

该代码实现了一个基于 Gradio 的 YOLOv10 推理应用，支持以下功能：

* WebRTC 实时视频流推理：通过摄像头实时检测目标。

* 图片上传推理：上传图片并进行目标检测。

* 视频上传推理：上传视频并逐帧进行目标检测，显示推理结果。

## 代码结构

**1. 导入必要的库**

```python
import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import numpy as np
import time
import PIL.Image as Image
from collections import deque
from dotenv import load_dotenv
```


gradio：用于构建用户界面。

cv2：用于视频处理和图像操作。

ultralytics：用于加载 YOLOv10 模型并进行推理。

numpy：用于数组操作。

PIL.Image：用于处理图像格式。

deque：用于存储时间戳以计算 FPS。

dotenv：用于加载环境变量（如 Twilio 配置）。

**2. 加载环境变量**

```python
load_dotenv()
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
```

加载 .env 文件中的 Twilio 配置，用于 WebRTC 的 ICE 服务器配置。

如果未配置 Twilio，则 rtc_configuration 设置为 None。

**3. FPS 计算类**

```python
class FPS:
    def __init__(self, avg_frames=30):
        self.timestamps = deque(maxlen=avg_frames)
    
    def update(self):
        self.timestamps.append(time.time())
        
    def get(self):
        if len(self.timestamps) < 2:
            return 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
```

功能：计算实时帧率（FPS）。

方法：

update()：记录当前时间戳。

get()：计算最近帧的平均 FPS。

**4. 加载 YOLOv10 模型**

```python
model = YOLOv10("runs/V10train/exp3_fp16/weights/best.engine")
print(f'{model = }')
```

加载 YOLOv10 模型的 TensorRT 引擎文件，用于高效推理。

**5. 视频推理函数**

```python
def yolov10_inference_video(video_path):
    fps_counter = FPS()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter.update()
        results = model.predict(source=frame, imgsz=640, conf=0.05)
        r = results[0]
        im_bgr = r.plot()
        fps = fps_counter.get()
        cv2.putText(im_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        yield im_bgr

    cap.release()
```
功能：逐帧读取视频，进行目标检测，并返回推理结果。

步骤：

打开视频文件。

逐帧读取视频。

使用 YOLOv10 模型进行推理。

在每帧上绘制推理结果和 FPS 信息。

使用 yield 将每帧推理结果返回。

**6. 图片推理函数**

```python
def yolov10_inference(image):
    fps_counter.update()
    results = model.predict(source=image, imgsz=640, conf=0.05, save=False)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
    fps = fps_counter.get()
    print(f"FPS: {fps:.1f}")
    return im
```

功能：对上传的图片进行目标检测。

步骤：

更新 FPS 计数器。

使用 YOLOv10 模型进行推理。

返回推理结果图像。

**7. Gradio 应用界面**

```python
def app():
    with gr.Blocks():
        with gr.Tabs():
            with gr.TabItem("WebRTC Stream"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(sources=["webcam"], type="pil", streaming=True)
                    image.stream(
                        fn=yolov10_inference, inputs=[image], outputs=[image], stream_every=0.1
                    )
            with gr.TabItem("Image Upload"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        output_image = gr.Image(label="Output Image", type="pil")
                    input_image.change(
                        fn=yolov10_inference, inputs=[input_image], outputs=[output_image]
                    )
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_output = gr.Video(label="Output Video")
                    video_input.change(
                        fn=lambda video_path: list(yolov10_inference_video(video_path)),
                        inputs=[video_input],
                        outputs=[video_output]
                    )
```

功能：构建 Gradio 用户界面。

选项卡：

WebRTC Stream：实时视频流推理。

Image Upload：上传图片并进行推理。

Video Upload：上传视频并逐帧显示推理结果。


**8. 启动应用**

```python
gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML("<h1 style='text-align: center'>YOLOv10: Real-Time End-to-End Object Detection</h1>")
    gr.HTML("<h3 style='text-align: center'><a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a></h3>")
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch(server_name='0.0.0.0')
```

功能：启动 Gradio 应用。


运行脚本，访问 http://0.0.0.0:7860。

功能选项：

WebRTC Stream：通过摄像头实时检测目标。

Image Upload：上传图片并查看检测结果。

Video Upload：上传视频并逐帧查看检测结果。









