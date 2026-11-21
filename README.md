# video-chroma-key

## 题面

绿幕抠像一开，暗部也被抠掉、人物边缘还挂着一圈绿边：① 距离用 RGB 而不是 YCbCr，暗部误抠；② 色度不居中（不减 0.5），距离整体偏大；③ alpha 只用单阈值，边缘硬；④ smoothness 不参与，两段式退化；⑤ 溢色抑制降所有通道，人物变色；⑥ 羽化在内部也做，细节丢失；⑦ 合成不预乘，边缘发黑；⑧ 参数不校验，smoothness < similarity 时除零；⑨ 批量共用 alpha 缓冲，帧之间串；⑩ 键色为黑时特判缺失，全黑画面被抠光。keying/chroma.py 是核心，tests/test_chroma.py 15个红。要修的是：BT.601 YCbCr 转换与色度居中、YCbCr 平面距离、两段式 alpha、仅键色通道的溢色抑制、只作用于软边缘的羽化、正确的预乘合成、参数校验与黑键拒绝、批量 alpha 隔离与原子回滚。

## 运行方式

```powershell
cd D:\gsb-20261119\video-chroma-key
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `keying/chroma.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_chroma.py` 是评测基准，**不可修改**；只允许修改 `keying/chroma.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数。
