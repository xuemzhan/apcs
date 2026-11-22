# Cache Translation Across Model Scale

《Cache Translation Across Model Scale: An Audit with Identity Controls and
Teacher-Content Probes》的论文、代码与逐样本数据。

跨模型 KV cache 翻译已经有多条技术路线——闭式线性映射、行为导向训练、离线蒸馏、选择性
重算——各自的成功标准也不相同。本文不提出新的 translator，而是给出一套**与具体 mapper
无关的三层审计**，让三个可以各自失败的问题分别可以打分：注入管线对不对、mapper 能不能在
替换级取代接收端自己的 prefill、教师缓存里有没有冻结学生可用的内容。

## 论文

| 版本 | 位置 | 形态 |
|---|---|---|
| arXiv 完整稿 | `paper/arxiv/` | 15 页，署名；图 4、表 6；含 `arxiv_submission.zip` |
| 双盲稿 | `paper/iclr2026/` | 正文 9 页（ICLR 样式，作者块默认注释） |
| 中文对照稿 | `paper/zh/` | 15 页，与英文稿逐段对应 |

根目录另有两份可直接分发的成品（与 `paper/*/main.pdf` 同字节），文件名见仓库根目录的
两份 PDF。

## 审计做什么

| 假说 | 门 | 结果 |
|---|---|---|
| **H1 机制** | 把学生自己的缓存走完整注入链路，是否复现它自己的 prefill | 通过（逐样本 logit 余弦 ≥ 0.9998） |
| **H2 替换** | 翻译后的缓存能否在替换级取代学生自身 prefill，区间下界 > −ε | 受测的重建导向映射器未达到；弱学生既不建立退化也不建立非劣 |
| **H3 增益** | 教师衍生内容能否带来区间排除零的正向增量 | 没有任何探针配置建立显著增益 |

结论的作用域是**严格零 re-prefill**下的**重建导向与架构先验**映射器。行为导向训练、蒸馏、
选择性重算、循环状态搬运等方法不在受测范围内，附录 E 逐个说明差别。

## 使用

```bash
pip install -e .

# 提交前总校验：引用闭合、页数、打包、分发件、48 项数值门（纯 CPU，不需要模型）
python scripts/verify_submission.py

# 论文里每个数字的数值门（10^4 次 bootstrap）
python scripts/paper_consistency_check.py

# 单次审计评测（需要模型）
python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run
```

## 数据与复现

- 逐样本评分记录：`reports/runs/`（177 个评测 run，每个 run 带 git hash 与协议版本）
- 计数口径：`scripts/aggregate_claims.py`
- 哪些指标只靠已存分数就能重算、哪些必须 GPU、已知不可复现项：`REPRODUCIBILITY.md`

## 文档

- `REPRODUCIBILITY.md` —— 材料位置、重算命令、GPU 步骤
- `docs/protocol/` —— 实验协议与历轮实验终审表
- `docs/design/design.md` —— 设计规格
- `docs/audits/` —— 逐轮审稿意见与处理记录
- `paper/arxiv/README_SUBMIT.md` —— arXiv 元数据与纯文本摘要

## 作者

Xuemin Zhang、Liangbin Hu、Kun Yi、Liheng Zhong、Junpeng Yu（CETC）。
投稿用的双盲版本在 `paper/iclr2026/`。

## 引用

```bibtex
@misc{zhang2026cachetranslation,
  title  = {Cache Translation Across Model Scale: An Audit with Identity Controls
            and Teacher-Content Probes},
  author = {Zhang, Xuemin and Hu, Liangbin and Yi, Kun and Zhong, Liheng and Yu, Junpeng},
  year   = {2026}
}
```

## License

MIT，见 `LICENSE`。
