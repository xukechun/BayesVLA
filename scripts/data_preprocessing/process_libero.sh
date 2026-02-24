eval "$(conda shell.bash hook)"
conda activate libero

HOME_DIR=$(dirname($(dirname $(realpath $0))))

# first download dataset from:
# https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets


python data_prepare/process_libero.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir $HOME_DIR/data_raw/libero \
    --libero_target_dir $HOME_DIR/data_processed/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir $HOME_DIR/data_raw/libero \
    --libero_target_dir $HOME_DIR/data_processed/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir $HOME_DIR/data_raw/libero \
    --libero_target_dir $HOME_DIR/data_processed/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir $HOME_DIR/data_raw/libero \
    --libero_target_dir $HOME_DIR/data_processed/libero \
    --skip_saved \
    --visualize
