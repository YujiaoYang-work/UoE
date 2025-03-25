
## LRA Benchmark

We released the source code for LRA benchmark.

To prepare the datasets, one would need to download the source code from [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena` folder in folder `LRA/datasets/` and also download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder `LRA/datasets/`. The directory structure would be
```
LRA/datasets/long-range-arena
LRA/datasets/lra_release
```
Then, run `sh create_datasets.sh` and it will create train, dev, and test dataset pickle files for each task.

## Scripts

Then, create the path for the checkpoints: `mkdir LRA_chks`. Finally, simply execute
  ```angular2html
  bash run_text.sh
  ```
  ```angular2html
  bash run_listops.sh
  ```
  ```angular2html
  bash run_retrieval.sh
  ```
  ```angular2html
  bash run_image.sh
  ```
  ```angular2html
  bash run_pathfinder.sh
  ```
