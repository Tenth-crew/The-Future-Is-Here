export CUDA_VISIBLE_DEVICES=0
# 自适应模型数量

MODEL_NAMES="PatchTST,DLinear"
MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 1, "d_layers": 1}'
# meta model option: mlp, linear_model, weighted_average
meta_model='attention_fusion'

# MODEL_NAMES="PatchTST"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}'

# MODEL_NAMES="PatchTST, TimesNet"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}'

# MODEL_NAMES="PatchTST, TimesNet, DLinear"
# MODEL_PARAMS_JSON_LIST='{"e_layers": 1, "d_layers": 1, "n_heads": 2}|||{"e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32, "top_k": 5}|||{"e_layers": 1, "d_layers": 1}'

# python -u run.py \
#     --task_name stacking \
#     --is_training 1 \
#     --root_path ./dataset/react-naive/ \
#     --data_name "react-naive" \
#     --data_path react-naive_to2017end.csv \
#     --model_id "$meta_model"\
#     --model PatchTST_Dlinear \
#     --data custom \
#     --features M \
#     --seq_len 84 \
#     --label_len 84 \
#     --pred_len 84 \
#     --meta_hidden_dim 64 \
#     --factor 3 \
#     --enc_in 5 \
#     --dec_in 5 \
#     --c_out 5 \
#     --base_model_train_epochs 10 \
#     --meta_model_train_epochs 15 \
#     --base_model_patience 3 \
#     --meta_model_patience 5 \
#     --meta_learner_type "$meta_model" \
#     --result_path workresult/stacking/1_dataset_test/exp_"$meta_model"_2model_PatchTST_Dlinear_react_naive \
#     --use_context_aware \
#     --context_output_dim 32\
#     --softmax_temperature 1 \
#     --d_model 512 \
#     --n_heads 8 \
#     --do_exp1_case_study \
#     --do_exp2_weight_dist \
#     --base_model_names "$MODEL_NAMES" \
#     --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
#     --des 'Exp' \
#     --inverse \
#     --itr 1 | tee -a logs/stacking/stacking_test_1_dataset.log


python -u run.py \
    --task_name stacking \
    --is_training 1 \
    --root_path ./dataset/vue/ \
    --data_name "vue" \
    --data_path vue_to2017end.csv \
    --model_id "$meta_model"\
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
    --meta_learner_type "$meta_model" \
    --result_path workresult/stacking/1_dataset_test/exp_"$meta_model"_2model_PatchTST_Dlinear_vue \
    --use_context_aware \
    --context_output_dim 32\
    --softmax_temperature 1 \
    --d_model 512 \
    --n_heads 8 \
    --do_exp1_case_study \
    --do_exp2_weight_dist \
    --base_model_names "$MODEL_NAMES" \
    --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
    --des 'Exp' \
    --inverse \
    --itr 1 | tee -a logs/stacking/stacking_test_1_dataset.log


# python -u run.py \
#     --task_name stacking \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh2.csv \
#     --model_id stacking_test \
#     --model P_D \
#     --data ETTh2 \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --meta_hidden_dim 64 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --base_model_train_epochs 10 \
#     --meta_model_train_epochs 15 \
#     --base_model_patience 3 \
#     --meta_model_patience 5 \
#     --meta_learner_type "$meta_model" \
#     --result_path workresult/stacking/exp_"$meta_model"_independent_test \
#     --softmax_temperature 1 \
#     --base_model_names "$MODEL_NAMES" \
#     --base_model_params_json_list "$MODEL_PARAMS_JSON_LIST" \
#     --des 'Exp' \
#     --inverse \
#     --itr 1 | tee -a logs/stacking/stacking_indepentdent_test.log