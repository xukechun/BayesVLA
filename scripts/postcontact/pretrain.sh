#!/bin/bash

set -e  # Exit on error

# Default values
GPU_IDS="${GPU_IDS:-0}"
CONFIG_NAME="${CONFIG_NAME:-pretrain}"
VA_SAVE_NAME="${VA_SAVE_NAME:-bayesvla_postcontact_va_pretrain}"
VA_CONTI_NAME="${VA_CONTI_NAME:-}"  # Stage 0 continue checkpoint (optional)
MASTER_PORT="${MASTER_PORT:-29500}"
BS="${BS:-64}"
MAX_ITER="${MAX_ITER:-50000}"
STAGE="${STAGE:-0}"  # "0"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --va-name)
            VA_SAVE_NAME="$2"
            shift 2
            ;;
        --va-conti)
            VA_CONTI_NAME="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --bs)
            BS="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --gpus GPU_IDS              GPU IDs to use (e.g., '0,1' or '0,1,2,3') [default: $GPU_IDS]"
            echo "  --stage STAGE               Training stage: '0', '1', or 'both' [default: $STAGE]"
            echo "  --config CONFIG_NAME        Config name [default: $CONFIG_NAME]"
            echo "  --va-name NAME              Stage 0 checkpoint save name [default: $VA_SAVE_NAME]"
            echo "  --va-conti NAME             Stage 0 checkpoint to continue from (uses -c instead of -s)"
            echo "  --port PORT                 Master port [default: $MASTER_PORT]"
            echo "  --bs BATCH_SIZE             Batch size [default: $BS]"
            echo "  --max-iter ITERATIONS       Max iterations [default: $MAX_ITER]"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate number of GPUs from GPU_IDS
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Validate inputs
if [[ ! "$STAGE" =~ ^(0|1|both)$ ]]; then
    echo "Error: --stage must be '0', '1', or 'both'"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "  GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Config: $CONFIG_NAME"
echo "  Stage 0 name: $VA_SAVE_NAME"
if [[ -n "$VA_CONTI_NAME" ]]; then
    echo "  Stage 0 continue from: $VA_CONTI_NAME"
fi
echo "  Master port: $MASTER_PORT"
echo "  Batch size: $BS"
echo "  Max iterations: $MAX_ITER"
echo "  Stage(s) to run: $STAGE"
echo "=========================================="
echo ""

# Stage 0: Pretraining VA
if [[ "$STAGE" == "0" ]]; then
    echo "=========================================="
    echo "STAGE 0: Pretraining (VA model)"
    echo "=========================================="
    echo "Running on GPUs $GPU_IDS with $NUM_GPUS processes"
    
    # Determine whether to use -s (save) or -c (continue)
    if [[ -n "$VA_CONTI_NAME" ]]; then
        echo "Continuing from checkpoint: $VA_CONTI_NAME"
        # Use -c to continue from existing checkpoint
        STAGE0_CMD="-c $VA_CONTI_NAME"
    else
        echo "Starting new training, saving to: $VA_SAVE_NAME"
        # Use -s to save to new checkpoint directory
        STAGE0_CMD="-s $VA_SAVE_NAME"
    fi
    echo ""
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --master_port $MASTER_PORT \
        --nproc_per_node $NUM_GPUS \
        -m models.train \
        --config $CONFIG_NAME \
        $STAGE0_CMD \
        --contact_phase post \
        --train_stage 0 \
        --bs $BS \
        --max_iterations $MAX_ITER
    
    if [ $? -ne 0 ]; then
        echo "Error: Stage 0 training failed!"
        exit 1
    fi
    
    echo ""
    echo "Stage 0 completed successfully!"
    echo ""
fi