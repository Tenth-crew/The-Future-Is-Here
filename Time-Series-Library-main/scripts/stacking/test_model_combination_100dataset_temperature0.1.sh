#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 定义不同的模型组合
model_combinations=(
    "PatchTST,DLinear|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 1, \"d_layers\": 1}"
    "iTransformer,DLinear|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 128, \"d_ff\": 128}|||{\"e_layers\": 1, \"d_layers\": 1}"
    "PatchTST,FreTS|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}"
    "PatchTST,Autoformer|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 2, \"d_layers\": 1}"
    "iTransformer,TimesNet|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 128, \"d_ff\": 128}|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 16, \"d_ff\": 32, \"top_k\": 5}"
    "PatchTST,SegRNN|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"seg_len\": 28, \"d_model\": 512, \"dropout\": 0.5, \"learning_rate\": 0.0001}"
    "PatchTST,TimesNet,DLinear|||{\"e_layers\": 1, \"d_layers\": 1, \"n_heads\": 2}|||{\"e_layers\": 2, \"d_layers\": 1, \"d_model\": 16, \"d_ff\": 32, \"top_k\": 5}|||{\"e_layers\": 1, \"d_layers\": 1}"
)

# meta model option: mlp, linear_model, weighted_average（加权求和），equal_weighted(平均权重) , gating_network, attention_fusion, context_only_predictor
meta_model='attention_fusion'

# 定义目标文件夹路径
target_dir="dataset/random100_from_21-23-dataset"

# 初始化数组
repoName=()

# 提取 dataset 列（跳过标题行）
datasets=$(awk -F',' 'NR > 1 {print $1}' workresult/patchtst/result.csv)

# 遍历 datasets 并查找对应的文件
for dataset in $datasets; do
    # 构造文件名模式
    file_pattern="$target_dir/${dataset}_all_roll_ewma_span28_normalize.csv"
    
    # 检查文件是否存在
    if [ -f "$file_pattern" ]; then
        repoName+=("$dataset")
    else
        echo "Warning: File for dataset '$dataset' not found."
    fi
done

# 遍历每种模型组合
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
    
    # 生成模型名称（用下划线连接）
    model_name=$(echo "$MODEL_NAMES" | sed 's/ //g' | sed 's/,/_/g')
    
    echo "Running experiment with models: $MODEL_NAMES"
    
    # 输出匹配的 reponame 数组以供调试
    # echo "Matched repositories: ${repoName[@]}"
    # echo "匹配数量: ${#repoName[@]}"

    # 不同模型的配置记得修改model,result_path等参数
    for repo in "${repoName[@]}"; do
        python -u run.py \
        --task_name stacking \
        --is_training 1 \
        --root_path ./dataset/random100_from_21-23-dataset/ \
        --data_name "$repo" \
        --data_path "$repo"_all_roll_ewma_span28_normalize.csv \
        --model_id "$meta_model"\
        --model "$model_name" \
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
        --meta_learner_type "$meta_model" \
        --result_path workresult/stacking/temperature/0.1/exp_"$meta_model"_${model_name}_100dataset \
        --calculate_correlation \
        --use_context_aware \
        --d_model 512\
        --n_heads 8 \
        --context_output_dim 32\
        --softmax_temperature 0.1 \
        --base_model_names "$MODEL_NAMES" \
        --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
        --des 'Exp' \
        --inverse \
        --itr 1 | tee -a logs/stacking/stacking_test_100_${model_name}.log
    done
done