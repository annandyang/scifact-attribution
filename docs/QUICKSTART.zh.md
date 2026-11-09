# Windows 复跑说明

当前电脑已用 Python 3.11、PyTorch CUDA 12.6 在 RTX 4060 Laptop（8 GB）上验证 NF4 运算。命令在项目根目录的 PowerShell 里执行。

## 先打开演示

演示只读缓存。本仓库已保存这次实测的预测，首次打开前下载并准备官方数据，就能在演示中查看证据原文。安装 Python 3.11 后，在 PowerShell 执行：

    py -3.11 -m venv .venv
    $py = ".\.venv\Scripts\python.exe"
    & $py -m pip install -r requirements-demo.txt
    & $py scripts/01_prepare_data.py
    & $py app/gradio_app.py --split dev --port 7860

如果系统没有 py 命令，用 Python 3.11 的实际安装路径代替。完成安装后也可以双击 launch_demo.cmd。打开 http://127.0.0.1:7860，停止时在 PowerShell 按 Ctrl+C。端口被占用就换成 7861。

先选声明看 C0–C5 与证据句，再看批量结果，最后看回答率和风险。声明页可切换三档 α，批量表使用冻结的 α。NEI 是模型对证据不足的判断，ABSTAIN 是门控拒答。

## 重新计算

同一个 Python 3.11 环境中：

    & $py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    & $py -m pip install -r requirements-gpu.txt
    & $py scripts/00_check_env.py --stage gpu

首次运行需要联网下载权重。若已有模型缓存，可以把 HF_HOME 指向缓存目录，并设置 HF_HUB_OFFLINE=1。权重下载中断时可以重试。先确认模型能加载：

    .\run_windows.ps1 -Stage Smoke -Python $py

之后运行开发集实验：

    .\run_windows.ps1 -Stage Dev -Python $py -Seeds "0" -SkipE1

使用已有检索表时可用 -SkipE1，需要重做时去掉。完整三种子用 -Seeds "0,1,2"。如只补跑 C3，用 scripts/04_run_condition.py --condition c3 --seeds 0。脚本 05 会重新训练指定种子；同名适配器和结果会被覆盖，改动实验设置前先保留旧结果。正式结果与短跑各自保存。

    .\run_windows.ps1 -Stage Test -Python $py -Seeds "0"
    .\run_windows.ps1 -Stage Ablation -Python $py -Seeds "0"
    & $py scripts/09_evidence_perturb.py --seed 0
    & $py scripts/10_error_review.py --seed 0

Test 只输出预测。证据扰动脚本先用 --limit 8 检查耗时。人工复核表里的判断列需要你阅读证据后填写。耗时实验请分开运行，避免多个程序争用显存。

需要从新结果重画 PNG 时，给该 Python 环境安装 matplotlib，再运行 scripts/12_plot_results.py。

## 从哪里读代码

先看 configs/conditions.yaml，再看 src/scifact_attr/system.py 的 run_claim，它把检索、选句、读者和结果行连起来。脚本 06 把 C4 转成 C5，脚本 08 从逐条结果生成表格。

最终 C4 先用同一个 Qwen 的基础模型写一条证据说明，再用 C3 的 LoRA 适配器判标签。说明中的引用必须对应输入句子。完整运行 ZIP 的 outputs/pilots/direct_cot 保存了初版直接生成的原始输出。

答辩可参考课件 IV–VI 的 PDF 第 5、12 页的 LoRA 与 QLoRA，第 37 页的两阶段推理，以及第 49–53 页的 RAG。六组对照、空证据捷径和拒答校准是项目自己的实验设计。

如果 Windows 报页面文件不足，先停下本项目其他 Python 进程，串行执行训练和推理。中断前已保存过一个训练轮次时，可使用脚本 05 的 --resume 参数，种子和训练配置需与中断前一致。
