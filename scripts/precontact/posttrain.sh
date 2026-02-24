#!/bin/bash

set -e  # Exit on error

# Default values
GPU_IDS="${GPU_IDS:-0}"
CONFIG_NAME="${CONFIG_NAME:-finetune_pp_arti}"   
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
VLA_SAVE_NAME="${VLA_SAVE_NAME:-bayesvla_precontact_vla_finetune}"
VLA_CONTI_NAME="${VLA_CONTI_NAME:-}"  # Stage 1 continue checkpoint (optional)
MASTER_PORT="${MASTER_PORT:-29500}"
BS="${BS:-64}"
MAX_ITER="${MAX_ITER:-25000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
STAGE="${STAGE:-1}"

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
        --pretrained-ckpt)
            PRETRAINED_CKPT="$2"
            shift 2
            ;;
        --vla-name)
            VLA_SAVE_NAME="$2"
            shift 2
            ;;
        --vla-conti)
            VLA_CONTI_NAME="$2"
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
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --gpus GPU_IDS                          GPU IDs to use (e.g., '0,1' or '0,1,2,3') [default: $GPU_IDS]"
            echo "  --stage STAGE                           Training stage: '0', '1', or 'both' [default: $STAGE]"
            echo "  --config CONFIG_NAME                    Config name [default: $CONFIG_NAME]"
            echo "  --pretrained-ckpt PRETRAINED_CKPT       Pretrained checkpoint path [default: $PRETRAINED_CKPT]"
            echo "  --vla-name NAME                         Stage 1 checkpoint save name [default: $VLA_SAVE_NAME]"
            echo "  --vla-conti NAME                        Stage 1 checkpoint to continue from (uses -c instead of -s)"
            echo "  --port PORT                             Master port [default: $MASTER_PORT]"
            echo "  --bs BATCH_SIZE                         Batch size [default: $BS]"
            echo "  --max-iter ITERATIONS                   Max iterations [default: $MAX_ITER]"
            echo "  --save-interval SAVE_INTERVAL           Save interval [default: $SAVE_INTERVAL]"
            echo "  -h, --help                              Show this help message"
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
echo "  Pretrained checkpoint: $PRETRAINED_CKPT"
echo "  Stage 1 name: $VLA_SAVE_NAME"
if [[ -n "$VLA_CONTI_NAME" ]]; then
    echo "  Stage 1 continue from: $VLA_CONTI_NAME"
fi
echo "  Master port: $MASTER_PORT"
echo "  Batch size: $BS"
echo "  Max iterations: $MAX_ITER"
echo "  Stage(s) to run: $STAGE"
echo "  Save interval: $SAVE_INTERVAL"
echo "=========================================="
echo ""

# Stage 1: Finetuning VLA (stage 0 is not necessarily required)
if [[ "$STAGE" == "1" ]]; then
    echo "=========================================="
    echo "STAGE 1: Finetuning (VLA model)"
    echo "=========================================="
    echo "Running on GPUs $GPU_IDS with $NUM_GPUS processes"
    
    
    # Determine whether to use -s (save) or -c (continue) for stage 1
    if [[ -n "$VLA_CONTI_NAME" ]]; then
        echo "Continuing from Stage 1 checkpoint: $VLA_CONTI_NAME"
        # Use -c to continue from existing stage 1 checkpoint
        STAGE1_CMD="-c $VLA_CONTI_NAME"
    else
        echo "Starting new Stage 1 training, saving to: $VLA_SAVE_NAME"

        if [[ -n "$PRETRAINED_CKPT" ]]; then
            STAGE1_CMD="--pretrained_ckpt $PRETRAINED_CKPT -s $VLA_SAVE_NAME"
            echo "Adapting from pretrained VLA checkpoint: $PRETRAINED_CKPT"
        else
            STAGE1_CMD="-s $VLA_SAVE_NAME"
        fi

    fi
    echo ""
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --master_port $MASTER_PORT \
        --nproc_per_node $NUM_GPUS \
        -m models.train \
        --config $CONFIG_NAME \
        $STAGE1_CMD \
        --contact_phase pre \
        --train_stage 1 \
        --bs $BS \
        --max_iterations $MAX_ITER \
        --save_interval $SAVE_INTERVAL
    
    if [ $? -ne 0 ]; then
        echo "Error: Stage 1 training failed!"
        exit 1
    fi
    
    echo ""
    echo "Stage 1 completed successfully!"
    echo ""
fi