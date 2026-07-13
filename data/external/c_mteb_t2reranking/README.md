---
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev-*
dataset_info:
  features:
  - name: query
    dtype: string
  - name: positive
    sequence: string
  - name: negative
    sequence: string
  splits:
  - name: dev
    num_bytes: 206865573
    num_examples: 6129
  download_size: 120293598
  dataset_size: 206865573
---
# Dataset Card for "T2Reranking"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)