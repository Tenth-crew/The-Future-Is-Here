#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 这个和one_baseModel_compare.sh分开跑的原因是一会好看结果对比。

# model_combinations=(
#     "PatchTST,DLinear|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 1, \"d_layers\": 1}"
#     "iTransformer,DLinear|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 128, \"d_ff\": 128}|||{\"e_layers\": 1, \"d_layers\": 1}"
#     "PatchTST,FreTS|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}"
#     "PatchTST,Autoformer|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 2, \"d_layers\": 1}"
#     "iTransformer,TimesNet|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 128, \"d_ff\": 128}|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 16, \"d_ff\": 32, \"top_k\": 5}"
#     "PatchTST,SegRNN|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"seg_len\": 28, \"d_model\": 512, \"dropout\": 0.5, \"learning_rate\": 0.0001}"
#     "PatchTST,TimesNet,DLinear|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 16, \"d_ff\": 32, \"top_k\": 5}|||{\"e_layers\": 1, \"d_layers\": 1}"
# )

model_combinations=(
    "PatchTST,DLinear|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 1, \"d_layers\": 1}"
)

meta_model='attention_fusion'

repoName=("vscode" "vue" "transformers" "kubernetes" "react-naive")

# 遍历每个模型
for combination in "${model_combinations[@]}"; do
    # 分离 MODEL_NAMES 和 MODEL_PARAMS_JSON_LIST
    IFS='|||' read -ra parts <<< "$combination"
    MODEL_NAMES="${parts[0]}"
    MODEL_PARAMS_JSON_LIST=""
    
    # 构建 MODEL_PARAMS_JSON_LIST
    for ((i=1; i<${#parts[@]}; i++)); do
        if [ $i -eq 1 ]; then
            MODEL_PARAMS_JSON_LIST="${parts[i]}"
        else
            MODEL_PARAMS_JSON_LIST="${MODEL_PARAMS_JSON_LIST}|||${parts[i]}"
        fi
    done
    
    for repo in "${repoName[@]}"; do
        python -u run.py \
        --task_name stacking \
        --is_training 1 \
        --root_path ./dataset/"$repo"/ \
        --data_name "$repo" \
        --data_path "$repo"_all_roll_ewma_span28_normalize.csv \
        --model_id attention_fusion \
        --model "$MODEL_NAME" \
        --data custom \
        --features M \
        --seq_len 84 \
        --label_len 84 \
        --pred_len 84 \
        --meta_hidden_dim 64 \
        --factor 3 \
        --enc_in 5 \
        --dec_in 5 \
        --c_out 5 \
        --base_model_train_epochs 10 \
        --meta_model_train_epochs 15 \
        --base_model_patience 3 \
        --meta_model_patience 5 \
        --meta_learner_type attention_fusion \
        --result_path workresult/stacking/big_repo_baseModel_combination_compare/exp_""$MODEL_NAMES""_"$repo" \
        --use_context_aware \
        --context_output_dim 32 \
        --softmax_temperature 1 \
        --d_model 512 \
        --n_heads 8 \
        --do_exp1_case_study \
        --do_exp2_weight_dist \
        --base_model_names "$MODEL_NAMES" \
        --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
        --des 'Exp' \
        --inverse \
        --itr 1 | tee -a logs/stacking/stacking_test_100.log
    done
done