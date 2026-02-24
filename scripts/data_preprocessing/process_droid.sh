eval "$(conda shell.bash hook)"
conda activate lerobot

HOME_DIR=$(dirname($(dirname $(realpath $0))))

# first download dataset from:
# https://huggingface.co/datasets/cadene/droid_1.0.1


python data_prepare/process_droid.py \
    --input_root $HOME_DIR/data_raw/droid_1.0.1 \
    --output_root $HOME_DIR/data_processed/droid \
    --skip_saved
