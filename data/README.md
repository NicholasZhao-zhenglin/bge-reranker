# 数据说明

当前实验使用 Hugging Face 的 `C-MTEB/T2Reranking` 数据集。

目录约定：

- `external/c_mteb_t2reranking/`：原始 Parquet 和数据卡；原始 Parquet 默认不提交到 Git。
- `experiments/c_mteb_t2reranking/`：固定数据划分和 baseline 评测产物；大文件默认不提交。

运行 `python scripts/evaluate_c_mteb_baseline.py` 会自动生成固定的 train/validation/test 划分。
