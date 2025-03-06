## Introduction

This directory contains the pytorch implementation of UoE's language modeling tasks. In this section, we conduct experiments on the Wikitext-103 dataset and the One Billion Word dataset.

## Data Prepration

```bash
bash getdata.sh
```

## Training and Evaluation

- Training on Wikitext-103 dataset

```bash
bash run_wt103.sh train --work_dir PATH_TO_WORK_DIR
```

- Evaluation on Wikitext-103 dataset

  `bash run_wt103.sh eval --work_dir PATH_TO_WORK_DIR`

- Training on One Billion Word dataset

  `bash run_lm1b.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation on One Billion Word dataset

  `bash run_lm1b.sh eval --work_dir PATH_TO_WORK_DIR`


- To see performance of other Transformer variants , You need to change the `attn_type` parameter. For example, `attn_type=8`  corresponds to DeepSeek-V3, `attn_type=6`  corresponds to XMoE
