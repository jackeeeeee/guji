# 车牌预测服务（plate-prediction-service）

一个可部署的车牌复检演示服务：前端选内置数据集、点开始预测，结果含耗时和预测有误列表（车牌图、预测值、真实值）。

## 它是什么、不是什么

- 是：HyperLPR3（小模型，ONNX Runtime CPU）做的车牌识别演示 + 试验脚本。
- 不是：生成式大模型 / LLM 做 OCR；也不是上线承诺。公开 CCPD 数据上的绿牌 90.0% 只是可行性信号（见 `TRIAL_REPORT.md`）。

## 快速启动（服务器上）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打开 http://服务器IP:8000
```

或用 Docker：

```bash
docker build -t plate-prediction-service .
docker run -p 8000:8000 plate-prediction-service
```

首次启动 HyperLPR3 会自动下载约 11.9MB 模型到 `~/.hyperlpr3`，服务器需能访问模型源。

## 前端用法

1. 打开首页，选数据集：绿牌 300 张 / 蓝牌 100 张 / 全部 400 张，无需上传。
2. 点“开始预测”：展示预计算结果（2026-11-03 夜间 CPU 实跑记录），含总耗时、平均耗时和预测有误列表。
3. 点“实时跑模型（前 50 张）”：服务器上真正调用 HyperLPR3。需要先把数据集放到 `data/`：
   - 绿牌：`data/CCPD2020/ccpd_green/test/*.jpg`
   - 蓝牌：`data/ccpd2019_partial/CCPD2019/ccpd_fn/*.jpg`
   - 详见 `scripts/download_ccpd.sh` 和 `TRIAL_REPORT.md`。没有数据集时该接口返回 409，并提示用预计算结果。

## 接口

- `GET /api/datasets`：内置数据集
- `GET /api/results?dataset=green|blue|all&errors_only=true|false`：预计算结果
- `POST /api/predict-live`：`{"dataset":"green","limit":50}`，实时跑模型
- `GET /api/image/<filename>`：样本图（仅随包附带的 30 张案例图，或已下载的数据集图）

## 目录

- `app/main.py`：FastAPI 服务
- `app/static/index.html`：前端（无构建步骤）
- `scripts/`：试验脚本（`run_hyperlpr3.py` 跑模型出 CSV，`ccpd_parse.py` 解析 CCPD 文件名标签，`evaluate.py` 从 CSV 复算准确率）
- `data/precomputed_*.csv`：2026-11-03 试水结果
- `data/samples/cases/`：对错案例图 30 张（绿牌对/错、蓝牌错各 10 张）
- `TRIAL_REPORT.md`：完整试水报告，含准确率、错误拆解、PaddleOCR 未跑通、LPRNet 未试等如实记录

## 部署给其他 agent 的注意事项

- 不需要 GPU 也能跑演示；实时大批量预测建议用 GPU 或多核 CPU，并先做小样本压测。
- 大数据集（CCPD2020 约 908MB、CCPD2019 约 13GB）**不随代码包附带**，`.gitignore` 已排除，服务器上按上文单独下载。
- 不要把真实卡口数据、API Key、生产配置放进这个仓库。
