#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_NAMES="PatchTST,DLinear"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1}'

meta_model=("mlp" "attention_fusion" "linear_model" "weighted_average" "equal_weighted" "gating_network")

repoName=("vscode" "vue" "transformers" "kubernetes" "react-naive")

# 遍历每个元模型和数据集
for meta_m in "${meta_model[@]}"; do
    for repo in "${repoName[@]}"; do
        python -u run.py \
        --task_name stacking \
        --is_training 1 \
        --root_path ./dataset/"$repo"/ \
        --data_name "$repo" \
        --data_path "$repo"_all_roll_ewma_span28_normalize.csv \
        --model_id "$meta_m" \
        --model PatchTST_Dlinear \
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
        --meta_learner_type "$meta_m" \
        --result_path workresult/stacking/big_repo_metaModel_compare/exp_"$meta_m"_"$repo"_2model_PatchTST_Dlinear_100dataset \
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