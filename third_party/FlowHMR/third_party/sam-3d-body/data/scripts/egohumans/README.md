# EgoHumans Dataset Preparation

Follow the steps below to prepare the **[EgoHumans Dataset](https://rawalkhirodkar.github.io/egohumans/)** for **[SAM-3D-Body Data](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)**.

---

- Set the following environment variables to simplify directory references:

    ```bash
    export EGOHUMANS_DATA_DIR=/path/to/egohumans/dataset
    export EGOHUMANS_IMG_DIR=/path/to/egohumans/undistorted/images
    ```

- Download 🔗 [EgoHumans Dataset](https://drive.google.com/drive/folders/1JD963urzuzV_R_6FOVOtlx8UupwUuknR) and extract all the files to `$EGOHUMANS_DATA_DIR`, following the directory structure below.

    ```plaintext
    $EGOHUMANS_DATA_DIR
    ├── 01_tagging
    │   ├── 001_tagging
    │   ├── 002_tagging
    │   └── ...
    ├── 02_lego
    │   ├── 001_legoassemble
    │   ├── 002_legoassemble
    │   └── ...
    └── ...
    ```

- Run the following command to undistort the EgoHumans images and save the results to `$EGOHUMANS_IMG_DIR`.

    ```bash
    python scripts/egohumans/undistort_egohumans.py \
        --src_dir $EGOHUMANS_DATA_DIR \
        --dst_dir $EGOHUMANS_IMG_DIR
    ```

- `$EGOHUMANS_IMG_DIR` should follow the directory structure below.

    ```plaintext
    $EGOHUMANS_IMG_DIR
    ├── 01_tagging
    │   ├── 001_tagging
    │   ├── 002_tagging
    │   └── ...
    ├── 02_lego
    │   ├── 001_legoassemble
    │   ├── 002_legoassemble
    │   └── ...
    └── ...
    ```
