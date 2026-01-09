export CUDA_VISIBLE_DEVICES=0

MODEL_NAMES="PatchTST,DLinear"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1}'

meta_model='gating_network'

# 定义目标文件夹路径
target_dir="dataset/random100_from_21-23-dataset"

# 定义seq_len参数列表
pred_lens="14 28 42 56 70 84 98 112 126 140"

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

# 外层循环遍历pred_len参数
for pred_len in $pred_lens; do
    echo "Running experiment with pred_len=$pred_len"
    
    for repo in "${repoName[@]}"; do
        python -u run.py \
        --task_name stacking \
        --is_training 1 \
        --root_path ./dataset/random100_from_21-23-dataset/ \
        --data_name "$repo" \
        --data_path "$repo"_all_roll_ewma_span28_normalize.csv \
        --model_id "$meta_model"\
        --model PatchTST_DLinear \
        --data custom \
        --features M \
        --seq_len 84 \
        --label_len 84 \
        --pred_len $pred_len \
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
        --result_path workresult/stacking/Parameter_Sensitivity_Analysis/pred_len/exp_"$meta_model"_2model_PatchTST_DLinear_predlen"$pred_len"_100dataset \
        --use_context_aware \
        --context_output_dim 32\
        --d_model 512\
        --n_heads 8 \
        --softmax_temperature 1 \
        --base_model_names "$MODEL_NAMES" \
        --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
        --des 'Exp' \
        --inverse \
        --itr 1 | tee -a logs/stacking/stacking_test_100_predlen"$pred_len".log
    done
done