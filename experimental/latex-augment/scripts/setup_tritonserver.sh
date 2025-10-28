# run within nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
#
# reference:
# https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/encoder_decoder.md

set -ex

cd /tmp
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.16.0 --depth 1
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive

cd /workspace
mkdir hf_models
cd hf_models
git clone https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt

export SERVING_DATA_PATH=/workspace/tritonserver_data
export MODEL_NAME=mbart-large-50-one-to-many-mmt
export MODEL_TYPE=bart
export HF_MODEL_PATH=/workspace/hf_models/${MODEL_NAME}
export UNIFIED_CKPT_PATH=$SERVING_DATA_PATH/ckpt/${MODEL_NAME}
export ENGINE_PATH=$SERVING_DATA_PATH/engines/${MODEL_NAME}
export MODEL_REPO_PATH=$SERVING_DATA_PATH/enc_dec_ifb
export INFERENCE_PRECISION=float16
export TP_SIZE=1
export MAX_BEAM_WIDTH=1
export MAX_BATCH_SIZE=64
export INPUT_LEN=1024
export OUTPUT_LEN=512

cd /tmp/tensorrtllm_backend
rm -rf $SERVING_DATA_PATH
python3 tensorrt_llm/examples/enc_dec/convert_checkpoint.py \
--model_type ${MODEL_TYPE} \
--model_dir ${HF_MODEL_PATH} \
--output_dir ${UNIFIED_CKPT_PATH} \
--dtype ${INFERENCE_PRECISION} \
--tp_size ${TP_SIZE}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/encoder \
--output_dir ${ENGINE_PATH}/encoder \
--kv_cache_type disabled \
--moe_plugin disable \
--max_beam_width ${MAX_BEAM_WIDTH} \
--max_input_len ${INPUT_LEN} \
--max_batch_size ${MAX_BATCH_SIZE} \
--gemm_plugin ${INFERENCE_PRECISION} \
--bert_attention_plugin ${INFERENCE_PRECISION} \
--gpt_attention_plugin ${INFERENCE_PRECISION} \
--remove_input_padding enable \
--log_level error

# mbart-50 requires output to start with EOS and LANG_ID tokens
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH}/decoder \
--output_dir ${ENGINE_PATH}/decoder \
--moe_plugin disable \
--max_beam_width ${MAX_BEAM_WIDTH} \
--max_batch_size ${MAX_BATCH_SIZE} \
--gemm_plugin ${INFERENCE_PRECISION} \
--bert_attention_plugin ${INFERENCE_PRECISION} \
--gpt_attention_plugin ${INFERENCE_PRECISION} \
--remove_input_padding enable \
--log_level error \
--max_input_len 2 \
--max_encoder_input_len ${INPUT_LEN} \
--max_seq_len ${OUTPUT_LEN}

cd /tmp/tensorrtllm_backend
rm -rf $MODEL_REPO_PATH
cp all_models/inflight_batcher_llm/ $MODEL_REPO_PATH -r
python3 tools/fill_template.py -i $MODEL_REPO_PATH/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:${MAX_BEAM_WIDTH},engine_dir:${ENGINE_PATH}/decoder,encoder_engine_dir:${ENGINE_PATH}/encoder,kv_cache_free_gpu_mem_fraction:0.8,cross_kv_cache_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,max_queue_size:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32
