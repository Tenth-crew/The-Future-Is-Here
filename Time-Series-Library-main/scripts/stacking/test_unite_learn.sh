export CUDA_VISIBLE_DEVICES=0

# model在stacking是不使用的参数，但是代码完整性要求有

# 统一参数脚本
# python -u run.py \
#     --task_name stacking \
#     --is_training 1 \
#     --root_path ./dataset/Top500/ \
#     --data_path transformers_all.csv \
#     --model_id stacking_test \
#     --model P_T_D \
#     --model_1 PatchTST \
#     --model_2 TimesNet \
#     --model_3 DLinear \
#     --data custom \
#     --features M \
#     --seq_len 84 \
#     --label_len 84 \
#     --pred_len 84 \
#     --e_layers 1 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 5 \
#     --dec_in 5 \
#     --c_out 5 \
#     --d_model 16 \
#     --d_ff 32 \
#     --top_k 5 \
#     --des 'Exp' \
#     --n_heads 2 \
#     --inverse \
#     --itr 1 | tee -a logs/stacking/stacking_test.log

# 各模型不同参数脚本，适用于固定三个模型的代码
# python -u run.py \
#     --task_name stacking \
#     --is_training 1 \
#     --root_path ./dataset/Top500/ \
#     --data_path vscode_all.csv \
#     --model_id stacking_test \
#     --model P_T_D \
#     --model_1 PatchTST \
#     --model_2 TimesNet \
#     --model_3 DLinear \
#     --data custom \
#     --features M \
#     --seq_len 84 \
#     --label_len 84 \
#     --pred_len 84 \
#     --factor 3 \
#     --enc_in 5 \
#     --dec_in 5 \
#     --c_out 5 \
#     --params_patchtst '{"e_layers": 1, "d_layers": 1, "n_heads": 2}' \
#     --params_timesnet '{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}' \
#     --params_dlinear '{"e_layers": 1, "d_layers": 1}' \
#     --des 'Exp' \
#     --inverse \
#     --itr 1 | tee -a logs/stacking/stacking_test.log

# 自适应模型数量

# MODEL_NAMES="PatchTST,DLinear"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1}'

MODEL_NAMES="PatchTST"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}'

# MODEL_NAMES="PatchTST, TimesNet"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}'

python -u run.py \
    --task_name stacking \
    --is_training 1 \
    --root_path ./dataset/language/Detail_repo/Python/ \
    --data_path transformers_all_roll_ewma_span28_normalize.csv \
    --model_id stacking_test \
    --model P_D \
    --data custom \
    --features M \
    --seq_len 84 \
    --label_len 84 \
    --pred_len 84 \
    --meta_hidden_dim 84 \
    --factor 3 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 5 \
    --base_model_names "$MODEL_NAMES" \
    --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
    --des 'Exp' \
    --inverse \
    --itr 1 | tee -a logs/stacking/stacking_unite_test.log

# python -u run.py \
#     --task_name stacking \
#     --is_training 1 \
#     --root_path ./dataset/language/Detail_repo/Python/ \
#     --data_path Python-100-Days_all_roll_ewma_span28_normalize.csv \
#     --model_id stacking_test \
#     --model P_D \
#     --data custom \
#     --features M \
#     --seq_len 84 \
#     --label_len 84 \
#     --pred_len 84 \
#     --meta_hidden_dim 84 \
#     --factor 3 \
#     --enc_in 5 \
#     --dec_in 5 \
#     --c_out 5 \
#     --base_model_names "$MODEL_NAMES" \
#     --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
#     --des 'Exp' \
#     --inverse \
#     --itr 1 | tee -a logs/stacking/stacking_test.log