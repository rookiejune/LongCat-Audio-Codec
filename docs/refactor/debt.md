# Refactor Debt

本文件记录 `third_party/LongCat-Audio-Codec` 这个本地 fork 中“能跑但设计边界还不够干净”的未完成重构债务。这里不评价上游代码风格本身，只记录本地后续维护时值得优先收敛的接口、边界和实现债务。

## Scope

使用本文件记录：

- Interface debt：公开入口、模块契约、配置面或调用方预期需要调整。
- Boundary debt：逻辑位于错误层级、运行时约束重复、脚本承载了可复用规则。
- Implementation debt：接口稳定但内部实现难维护，可以等下一次触碰时处理。

不要记录实验日志、benchmark 结果、私有路径、远程训练过程、checkpoint 或已完成历史。

## Priority

| Priority | Meaning |
|---|---|
| P0 | 当前阻塞开发、正确性或结果可信度。 |
| P1 | 近期大概率会被触碰，或问题正在扩散到其他模块。 |
| P2 | 可运行且相对稳定，等下次改到该区域时处理。 |
| P3 | 主要是审美问题，不主动排期。 |


## Debt Index

| ID | Type | Area | Problem | Risk | Priority | Status | Next Action | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | Boundary debt | Demo inference entry | `inference.py` 同时承担 CLI、音频 batch loader、模型加载、多 rate demo、token extraction demo 和输出写入。 | 作为 demo 可运行，但下游复用 codec API 时会复制 demo 内部逻辑，而不是调用稳定服务。 | P1 | pending | 把 encode/decode batch 逻辑抽入 package helper，`inference.py` 只保留 demo CLI 和打印。 | 证据：`inference.py:71-191`、`README.md:154-207`。 |
| R002 | Interface debt | Public loader / config contract | 安装包暴露 `longcat_audio_codec.load_encoder/load_decoder`，但底层仍依赖 `networks.semantic_codec.model_loader`、YAML 字段和相对 ckpt 路径；encoder 用 `strict=False` 加载。 | 公共 loader 对 checkpoint schema、缺失 key 和配置字段的错误边界不清楚；下游难以判断是兼容加载还是模型不匹配。 | P1 | pending | 定义 typed config/loader result，加载时显式报告 missing/unexpected keys；保留 YAML 入口但把 schema 校验集中。 | 证据：`longcat_audio_codec/model_loader.py:5-14`、`networks/semantic_codec/model_loader.py:57-78`、`networks/semantic_codec/model_loader.py:107-129`。 |
| R003 | Boundary debt | Checkpoint/resource path resolution | 本地 fork 增加了 `LONGCAT_AUDIO_CODEC_CKPT_DIR` 和 `HF_HOME` cache 解析，但 README 仍主要要求用户修改 YAML 或把 ckpt 放在项目根。 | 安装包模式、源码 demo 模式和 HF cache 模式都有入口，后续路径规则容易漂移。 | P2 | pending | 把 `resolve_checkpoint_path()` 和 `default_config_path()` 作为 README 的主路径契约，并让 demo 默认使用 package config。 | 证据：`README.md:95-135`、`longcat_audio_codec/paths.py:8-57`、`pyproject.toml:35-36`。 |
| R004 | Implementation debt | Streaming demo | `stream_inference_demo.py` 写死 config、示例 token、lookahead/buffer 常数和输出路径，只能作为一次性 demo。 | streaming 能力存在，但不能作为可复用 API 验证延迟、chunk 边界或不同 codebook 数。 | P2 | pending | 把 `stream_decode_tokens()` 提升为 package helper，demo 通过 CLI 接收 config/tokens/output，并补一个 synthetic decoder smoke。 | 证据：`stream_inference_demo.py:7-82`、`stream_inference_demo.py:87-148`。 |

## Acceptance

- `python -m unittest discover -s tests -v`
- `python -m compileall -q inference.py stream_inference_demo.py longcat_audio_codec networks semantic_tokenizer_general`
- `python inference.py --help`

