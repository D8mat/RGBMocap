# Harmony4D Dataset Preparation

Follow the steps below to prepare the **[Harmony4D Dataset](https://jyuntins.github.io/harmony4d/)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export HARMONY4D_DATA_DIR=/path/to/harmony4d/dataset
    export HARMONY4D_IMG_DIR=/path/to/harmony4d/undistorted/images
    ```

- Download 🔗 [Harmony4D Dataset](https://huggingface.co/datasets/Jyun-Ting/Harmony4D/tree/main) and extract all the files to `$HARMONY4D_DATA_DIR`, following the directory structure below.

    ```plaintext
    $HARMONY4D_DATA_DIR
    ├── test
    │   ├── 01_hugging
    │   │   └── 002_hugging
    │   ├── 03_grappling2
    │   │   ├── 025_grappling2
    │   │   └── ...
    │   ├── ...
    └── train
        ├── 01_hugging
        │   └── 001_hugging
        ├── 02_grappling
        │   ├── 001_grappling
        │   └── ...
        └── ...
    ```

- Run the following command to undistort the Harmony4D images and save the results to `$HARMONY4D_IMG_DIR`.

    ```bash
    python scripts/harmony4d/undistort_harmony4d.py \
        --src_dir $HARMONY4D_DATA_DIR \
        --dst_dir $HARMONY4D_IMG_DIR
    ```

- `$HARMONY4D_IMG_DIR` should the directory structure below.

    ```plaintext
    $HARMONY4D_IMG_DIR
    ├── test
    │   ├── 01_hugging
    │   │   └── 002_hugging
    │   ├── 03_grappling2
    │   │   ├── 025_grappling2
    │   │   └── ...
    │   ├── ...
    └── train
        ├── 01_hugging
        │   └── 001_hugging
        ├── 02_grappling
        │   ├── 001_grappling
        │   └── ...
        └── ...
    ```
