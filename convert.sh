set -e

SRCDIR=$(dirname "$0")

usage() {
    cat <<EOF
Usage: bash convert.sh <input_folder> [--name <name>] [--output <output_folder>]
                       [--complete] [--test-vocab] [--transformers-fp8]

  input_folder         HF-format model directory to convert
  --name NAME          basename used for output files (default: basename of input_folder)
  --output DIR         output directory (default: <input_folder>-GGUF)
  --complete           full build (all standard + dynamic quants), sets COMPLETE=1
  --test-vocab         only build a vocab-only GGUF and exit, sets TEST_VOCAB=1
  --transformers-fp8   also run convert_hf_to_fp8.py; FP8 output goes to
                       <input_folder>-FP8, or <output_folder>/FP8 if --output is set
EOF
    exit 1
}

INPUT_PATH=""
NAME=""
OUTPUT_PATH=""
OUTPUT_SPECIFIED=0
RUN_FP8=0

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)           usage ;;
        --name)              NAME=$2;                              shift 2 ;;
        --output)            OUTPUT_PATH=$2; OUTPUT_SPECIFIED=1;   shift 2 ;;
        --complete)          COMPLETE=1;                           shift ;;
        --test-vocab)        TEST_VOCAB=1;                         shift ;;
        --transformers-fp8)  RUN_FP8=1;                            shift ;;
        --)                  shift;                                break ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *)
            if [ -z "$INPUT_PATH" ]; then
                INPUT_PATH=$1
                shift
            else
                echo "unexpected positional argument: $1" >&2
                usage
            fi
            ;;
    esac
done

[ -z "$INPUT_PATH" ] && usage
[ -d "$INPUT_PATH" ] || { echo "input folder not found: $INPUT_PATH" >&2; exit 1; }

INPUT_PATH=${INPUT_PATH%/}
: "${NAME:=$(basename "$INPUT_PATH")}"
: "${OUTPUT_PATH:=${INPUT_PATH}-GGUF}"

# Flags ------------------------------------------------------------------
# TEST_VOCAB=1          : only build a vocab-only GGUF and exit (sanity check)
# COMPLETE=0            : minimal build (F16 base + Q4_K_M)
# COMPLETE=1            : full build (BF16 + F16 base, all standard quants,
#                         plus unsloth-style "dynamic" _L / _XL variants)
# STRIP_CHAT_TEMPLATE=0 : drop tokenizer.chat_template from the converted
#                         GGUF. Experiment to see if Ollama's nemotron_h
#                         hardcoded-template path can be sidestepped (lets the
#                         Modelfile TEMPLATE take effect). WARNING: llama-server
#                         loses /apply-template and /v1/chat/completions support.
# CLI --test-vocab / --complete override these; the assignments here are just defaults.
: "${TEST_VOCAB:=0}"
: "${COMPLETE:=0}"
STRIP_CHAT_TEMPLATE=0

# Base type used as source for all quantizations.
# F16 is the de-facto base; BF16 is safer for models trained in bf16.
BASE_TYPE="bf16"

# Calibration text for importance matrix (imatrix).
# Required for IQ1_*, IQ2_*, IQ3_XXS and recommended for all other quants
# (matches unsloth's "Dynamic Quants 2.0" recipe). Set to "" to disable.
IMATRIX_DATA="$SRCDIR/calibration.txt"

# Quants that REQUIRE an imatrix (will be skipped if none is available).
IMATRIX_REQUIRED_QUANTS=(IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS)

needs_imatrix() {
    local q=$1
    for r in "${IMATRIX_REQUIRED_QUANTS[@]}"; do
        [ "$q" = "$r" ] && return 0
    done
    return 1
}

# Quant types — matches unsloth/Llama-3.3-70B-Instruct-GGUF exactly.
# Other types supported by llama-quantize are listed below (commented out).
QUANTS_STD=(
    Q4_K_M
    Q5_K_M
    Q8_0
    # Q6_K
    # Q5_K_S
    # Q4_K_S
    # Q4_0   Q4_1
    # Q3_K_M Q3_K_S
    # Q2_K
    # IQ4_NL IQ4_XS
    # IQ3_XXS
    # IQ2_M  IQ2_XXS
    # IQ1_M  IQ1_S
    # Q3_K_L                # extra K-quant tier (between Q3_K_M and Q4_K_S)
    # Q2_K_S                # smaller Q2_K variant
    # Q5_0   Q5_1           # legacy 5-bit quants (superseded by Q5_K_*)
    # IQ3_M  IQ3_S  IQ3_XS  # extra IQ3 tiers
    # IQ2_S  IQ2_XS         # extra IQ2 tiers
    # TQ1_0  TQ2_0          # ternary quants (~1.7 / ~2.1 bpw, niche)
    # MXFP4_MOE             # MoE-only 4-bit
)

# Unsloth-style "dynamic" variants: keep token_embd and/or output at higher
# precision than the body. Format: "<NAME>:<TOK_TYPE>:<OUT_TYPE>".
# - _L  : token embeddings bumped to Q8_0
# - _XL : token embeddings AND output bumped to Q8_0 (or BF16 for Q8_K_XL)
QUANTS_DYNAMIC=(
    # "Q2_K_L:q8_0:"
    # "Q2_K_XL:q8_0:q8_0"
    # "Q3_K_XL:q8_0:q8_0"
    # "Q4_K_XL:q8_0:q8_0"
    # "Q5_K_XL:q8_0:q8_0"
    # "Q6_K_XL:q8_0:q8_0"
    # "Q8_K_XL:bf16:bf16"
)

# -----------------------------------------------------------------------

mkdir -p "$OUTPUT_PATH"
cd "$SRCDIR"

QUANTIZE=./build/bin/llama-quantize
IMATRIX_BIN=./build/bin/llama-imatrix

# Per-model imatrix path (set after $NAME / $OUTPUT_PATH are known).
IMATRIX=""

build_imatrix() {
    # Generate the importance matrix from $BASE_TYPE GGUF + calibration text.
    # No-op if disabled, already built, or calibration file missing.
    if [ -z "$IMATRIX_DATA" ]; then
        echo "[imatrix] disabled (IMATRIX_DATA empty)"
        IMATRIX=""
        return
    fi
    if [ ! -f "$IMATRIX_DATA" ]; then
        echo "[imatrix] WARNING: calibration file $IMATRIX_DATA not found — skipping imatrix"
        IMATRIX=""
        return
    fi
    IMATRIX=$OUTPUT_PATH/$NAME-imatrix.gguf
    if [ -f "$IMATRIX" ]; then
        echo "[skip] $IMATRIX already exists"
        return
    fi
    local base=$OUTPUT_PATH/$NAME-$(echo "$BASE_TYPE" | tr '[:lower:]' '[:upper:]').gguf
    CMD="$IMATRIX_BIN -m $base -f $IMATRIX_DATA -o $IMATRIX --process-output"
    echo "$CMD"
    eval $CMD
}

convert_hf() {
    # outtype is the CLI arg for --outtype (lowercase: bf16/f16/f32);
    # the filename uses the uppercase form (BF16/F16/F32) to match unsloth/bartowski.
    local outtype=$1
    local suffix=$(echo "$outtype" | tr '[:lower:]' '[:upper:]')
    local outfile=$OUTPUT_PATH/$NAME-$suffix.gguf
    if [ -f "$outfile" ]; then
        echo "[skip] $outfile already exists"
        return
    fi
    CMD="STRIP_CHAT_TEMPLATE=$STRIP_CHAT_TEMPLATE python3 convert_hf_to_gguf.py $INPUT_PATH --outfile $outfile --outtype $outtype"
    echo "$CMD"
    eval $CMD
}

quantize_std() {
    local quant=$1
    local base=$OUTPUT_PATH/$NAME-$(echo "$BASE_TYPE" | tr '[:lower:]' '[:upper:]').gguf
    local outfile=$OUTPUT_PATH/$NAME-$quant.gguf
    if [ -f "$outfile" ]; then
        echo "[skip] $outfile already exists"
        return
    fi
    local imat=""
    if [ -n "$IMATRIX" ] && [ -f "$IMATRIX" ]; then
        imat="--imatrix $IMATRIX"
    elif needs_imatrix "$quant"; then
        echo "[skip] $quant requires imatrix but none available"
        return
    fi
    CMD="$QUANTIZE $imat $base $outfile $quant"
    echo "$CMD"
    eval $CMD
}

quantize_dynamic() {
    # spec format: NAME:TOK_TYPE:OUT_TYPE  (TOK_TYPE or OUT_TYPE may be empty)
    local spec=$1
    local name=${spec%%:*}
    local rest=${spec#*:}
    local tok_type=${rest%%:*}
    local out_type=${rest#*:}

    # Derive the underlying base quant from the variant name:
    # Q2_K_L -> Q2_K, Q4_K_XL -> Q4_K, Q8_K_XL -> Q8_0 (special)
    local base_quant=${name%_XL}
    base_quant=${base_quant%_L}
    [ "$name" = "Q8_K_XL" ] && base_quant=Q8_0

    local base=$OUTPUT_PATH/$NAME-$(echo "$BASE_TYPE" | tr '[:lower:]' '[:upper:]').gguf
    local outfile=$OUTPUT_PATH/$NAME-$name.gguf
    if [ -f "$outfile" ]; then
        echo "[skip] $outfile already exists"
        return
    fi
    local flags=""
    [ -n "$tok_type" ] && flags="$flags --token-embedding-type $tok_type"
    [ -n "$out_type" ] && flags="$flags --output-tensor-type $out_type"
    if [ -n "$IMATRIX" ] && [ -f "$IMATRIX" ]; then
        flags="$flags --imatrix $IMATRIX"
    elif needs_imatrix "$base_quant"; then
        echo "[skip] $name (base $base_quant) requires imatrix but none available"
        return
    fi
    CMD="$QUANTIZE $flags $base $outfile $base_quant"
    echo "$CMD"
    eval $CMD
}

# -----------------------------------------------------------------------

if [ $TEST_VOCAB -eq 1 ]; then
    CMD="python3 convert_hf_to_gguf.py $INPUT_PATH --outfile $OUTPUT_PATH/$NAME-vocab.gguf --vocab-only"
    echo "$CMD"
    eval $CMD
    exit 0
fi

# Always build the base precision used for quantization.
convert_hf $BASE_TYPE

# Build the importance matrix from the base GGUF (needed for IQ quants,
# recommended for all quants). Falls back gracefully if disabled / data missing.
build_imatrix

if [ $COMPLETE -eq 1 ]; then
    # Other precision bases — uncomment to publish them too (unsloth ships
    # both BF16 and F16; F32 is rarely useful and ~4x the size of BF16/F16).
    # [ "$BASE_TYPE" = "bf16" ] && convert_hf f16
    # [ "$BASE_TYPE" = "f16" ]  && convert_hf bf16
    # convert_hf f32

    # Standard quants
    for q in "${QUANTS_STD[@]}"; do
        quantize_std $q
    done

    # Dynamic (unsloth-style) variants
    for spec in "${QUANTS_DYNAMIC[@]}"; do
        quantize_dynamic $spec
    done
else
    # Minimal build: one quant
    quantize_std Q4_K_M
fi

# Copy model-card assets into the output folder, substituting <name> -> $NAME
# in the two text templates. Binary assets (logos, etc.) are copied verbatim.
ASSETS_DIR="$SRCDIR/hf_assets_gguf"
if [ -d "$ASSETS_DIR" ]; then
    for src in "$ASSETS_DIR"/*; do
        [ -e "$src" ] || continue
        dst="$OUTPUT_PATH/$(basename "$src")"
        case "$(basename "$src")" in
            Modelfile|README.md)
                sed "s|<name>|$NAME|g" "$src" > "$dst"
                echo "[assets] wrote $dst (with <name> -> $NAME)"
                ;;
            *)
                cp -f "$src" "$dst"
                echo "[assets] copied $dst"
                ;;
        esac
    done
fi

if [ $RUN_FP8 -eq 1 ]; then
    if [ $OUTPUT_SPECIFIED -eq 1 ]; then
        FP8_PATH=$OUTPUT_PATH/FP8
    else
        FP8_PATH=${INPUT_PATH}-FP8
    fi
    mkdir -p "$FP8_PATH"
    # FP8_DYNAMIC is data-free and weight-only — no CUDA needed. Hide the GPU
    # so llmcompressor's data_free pipeline (which calls compressed_tensors'
    # dispatch_model → get_device_memory → torch.accelerator.get_memory_info)
    # doesn't trigger a CUDA context init that OOMs on unified-memory systems
    # (DGX Spark) when other processes are already holding most of the pool.
    CMD="CUDA_VISIBLE_DEVICES= python3 convert_hf_to_fp8.py $INPUT_PATH --outfile $FP8_PATH --split-max-size 5G --device cpu"
    echo "$CMD"
    eval $CMD
fi


# done
