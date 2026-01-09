#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_NAMES="PatchTST,DLinear"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1}'

meta_model='attention_fusion'

# 遍历标准数据集，格式为 "folder_name:data_file:dimensions:batch_size:other_params"
standard_datasets=(
    "electricity:electricity.csv:321:16:"
    "exchange_rate:exchange_rate.csv:8:32:"
    "illness:national_illness.csv:7:32:"
    "traffic:traffic.csv:862:8:"
    "weather:weather.csv:21:32:"
)

# 遍历标准数据集
for dataset_info in "${standard_datasets[@]}"; do
    # 分割字符串获取各个参数
    IFS=':' read -r folder_name data_file dimensions batch_size extra_params <<< "$dataset_info"
    
    data_name="${data_file%.*}"  # 移除文件扩展名获取数据集名称
    
    echo "Running experiment with dataset: $folder_name/$data_file"
    
    # 设置默认参数
    seq_len=96
    label_len=48
    pred_len=96
    d_ff=2048
    top_k=5
    
    # 根据数据集调整特定参数
    case "$folder_name" in
        traffic)
            d_ff=512
            top_k=5
            ;;
    esac
    
    python -u run.py \
    --task_name stacking \
    --is_training 1 \
    --root_path ./dataset/"$folder_name"/ \
    --data_path "$data_file" \
    --data_name "$data_name" \
    --model_id "$meta_model" \
    --model PatchTST_Dlinear \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --meta_hidden_dim 64 \
    --factor 3 \
    --enc_in $dimensions \
    --dec_in $dimensions \
    --c_out $dimensions \
    --base_model_train_epochs 10 \
    --meta_model_train_epochs 15 \
    --base_model_patience 3 \
    --meta_model_patience 5 \
    --meta_learner_type "$meta_model" \
    --result_path workresult/stacking/generalization_baseline/exp_"$meta_model"_${folder_name}_${data_name} \
    --use_context_aware \
    --context_output_dim 32 \
    --softmax_temperature 1 \
    --d_model 512 \
    --n_heads 8 \
    --d_ff $d_ff \
    --top_k $top_k \
    --base_model_names "$MODEL_NAMES" \
    --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
    --des 'Exp' \
    --inverse \
    --itr 1 | tee -a logs/stacking/generalization_baseline_${folder_name}.log
done

# 处理ETT-small数据集
ett_datasets=(
    "ETTh1.csv:32"
    "ETTh2.csv:32"
)

for dataset_info in "${ett_datasets[@]}"; do
    # 分割字符串获取文件名和batch_size
    IFS=':' read -r data_file batch_size <<< "$dataset_info"
    
    data_name="${data_file%.*}"  # 移除文件扩展名获取数据集名称
    folder_name="ETT-small"
    
    echo "Running experiment with dataset: ETT-small/$data_file"
    
    python -u run.py \
    --task_name stacking \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path "$data_file" \
    --data_name "$data_name" \
    --model_id "$meta_model" \
    --model PatchTST_Dlinear \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size $batch_size \
    --meta_hidden_dim 64 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --base_model_train_epochs 10 \
    --meta_model_train_epochs 15 \
    --base_model_patience 3 \
    --meta_model_patience 5 \
    --meta_learner_type "$meta_model" \
    --result_path workresult/stacking/generalization_baseline/exp_"$meta_model"_ETT-small_${data_name} \
    --use_context_aware \
    --context_output_dim 32 \
    --softmax_temperature 1 \
    --d_model 512 \
    --n_heads 8 \
    --base_model_names "$MODEL_NAMES" \
    --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
    --des 'Exp' \
    --inverse \
    --itr 1 | tee -a logs/stacking/generalization_baseline_ETT-small_${data_name}.log
done

echo "All experiments completed!"