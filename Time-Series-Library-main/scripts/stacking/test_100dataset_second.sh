export CUDA_VISIBLE_DEVICES=0
# 自适应模型数量

# MODEL_NAMES="PatchTST,DLinear"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2, "patch_len": 7, "stride": 7}|||{"e_layers": 1, "d_layers": 1}'

MODEL_NAMES="PatchTST, TimesNet, DLinear"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}|||{"e_layers": 1, "d_layers": 1}'

# MODEL_NAMES="PatchTST"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}'

# MODEL_NAMES="iTransformer, DLinear"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 2, "d_layers": 1, "d_model": 128, "d_ff": 128}|||{"e_layers": 1, "d_layers": 1}'

# MODEL_NAMES="PatchTST, FreTS"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1, "n_heads": 2}'

# MODEL_NAMES="PatchTST, Autoformer"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 2, "d_layers": 1}'

# MODEL_NAMES="iTransformer, TimesNet"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 2, "d_layers": 1, "d_model": 128, "d_ff": 128}|||{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}'

# MODEL_NAMES="PatchTST, SegRNN"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"seg_len": 28, "d_model": 512, "dropout": 0.5, "learning_rate": 0.0001}'



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
    --model PatchTST_Dlinear_TimesNet \
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
    --result_path workresult/stacking/compare_meta_model/exp_"$meta_model"_3model_PatchTST_Dlinear_TimesNet_temperature0.1_100dataset \
    --use_context_aware \
    --context_output_dim 32\
    --softmax_temperature 0.1 \
    --d_model 512 \
    --n_heads 8 \
    --base_model_names "$MODEL_NAMES" \
    --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
    --des 'Exp' \
    --inverse \
    --itr 1 | tee -a logs/stacking/stacking_test_100.log
done