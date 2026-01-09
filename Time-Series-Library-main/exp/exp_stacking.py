import json
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset # 用于元学习器数据
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr # 导入 pearson 相关系数计算函数
import matplotlib.pyplot as plt
from datetime import datetime
from models.GatingNetWork import GatingNetwork, GatedFusionModel, ContextOnlyPredictor  # 新增导入
from models.attention_fusion import Model as AttentionFusionModel # <-- 新增这行，使用别名避免冲突

# 确保这些 import 路径正确
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
# from utils.dtw_metric import accelerated_dtw # 如果 test 函数中要用 dtw
# from utils.augmentation import run_augmentation,run_augmentation_single # 如果需要

warnings.filterwarnings('ignore')


class Exp_stacking(Exp_Basic):
    def __init__(self, args):
        if not hasattr(args, 'do_exp1_case_study'):
            args.do_exp1_case_study = False # 默认关闭实验一
        if not hasattr(args, 'exp1_samples_to_visualize'):
            # 需要可视化的样本索引，以逗号分隔的字符串
            args.exp1_samples_to_visualize = "0,1,2" 
        if not hasattr(args, 'do_exp2_weight_dist'):
            args.do_exp2_weight_dist = False # 默认关闭实验二
        if not hasattr(args, 'meta_hidden_dim'): 
            args.meta_hidden_dim = 64
        if not hasattr(args, 'meta_dropout_rate'): 
            args.meta_dropout_rate = 0.0
        # 为基模型和元模型分别设置 epochs 和 patience (如果需要在args中区分)
        if not hasattr(args, 'base_model_train_epochs'):
            args.base_model_train_epochs = args.train_epochs # 默认与主训练epochs一致
        if not hasattr(args, 'meta_model_train_epochs'):
            args.meta_model_train_epochs = args.train_epochs # 默认与主训练epochs一致
        if not hasattr(args, 'base_model_patience'):
            args.base_model_patience = args.patience
        if not hasattr(args, 'meta_model_patience'):
            args.meta_model_patience = args.patience
        if not hasattr(args, 'meta_learner_type'):
            args.meta_learner_type = 'weighted_average' # 默认为新的加权平均方案
         # 为 softmax 温度参数设置默认值 (可选)
        if not hasattr(args, 'softmax_temperature'):
            args.softmax_temperature = 1.0
        if not hasattr(args, 'use_context_aware'):
            args.use_context_aware = False # 默认关闭上下文感知模块
        if not hasattr(args, 'context_output_dim'):
            args.context_output_dim = 32 # 默认编码器输出32维

        super(Exp_stacking, self).__init__(args)
        # 用于存储计算出的权重
        self.meta_weights = None

    def _override_args_for_base_model(self, shared_args, model_specific_params_dict):
        final_params_dict = vars(shared_args).copy()
        final_params_dict.update(model_specific_params_dict)
        return SimpleNamespace(**final_params_dict)


    def _build_model(self):
        args = self.args
        self.base_models = nn.ModuleList()

        if not hasattr(args, 'base_model_names') or not isinstance(args.base_model_names, list):
            raise ValueError("参数 'args.base_model_names' 必须是一个列表。")
        if not hasattr(args, 'base_model_params_json_list') or not isinstance(args.base_model_params_json_list, list):
            raise ValueError("参数 'args.base_model_params_json_list' 必须是一个列表。")
        if len(args.base_model_names) != len(args.base_model_params_json_list):
            raise ValueError("'base_model_names' 和 'base_model_params_json_list' 的长度必须匹配。")

        print(f"计划构建的基模型名称: {args.base_model_names}")
        for i, model_name in enumerate(args.base_model_names):
            model_custom_params_json = args.base_model_params_json_list[i]
            model_custom_params = json.loads(model_custom_params_json) if model_custom_params_json else {}
            current_model_args = self._override_args_for_base_model(args, model_custom_params)
            
            if model_name not in self.model_dict: # self.model_dict 来自 Exp_Basic
                raise ValueError(f"模型名称 '{model_name}' 在 model_dict 中未找到。可用模型: {list(self.model_dict.keys())}")

            print(f"构建基模型: {model_name}")
            model_instance = self.model_dict[model_name].Model(current_model_args).to(self.device)
            self.base_models.append(model_instance)
        
        print(f"实际构建的基模型数量: {len(self.base_models)}")
        if len(self.base_models) == 0:
            raise ValueError("没有指定或初始化任何基模型，无法进行stacking。")

        if args.features == 'MS':
            num_features_out_per_base_model = 1
            target_feature_dim = 1
        elif args.features == 'M':
            num_features_out_per_base_model = args.c_out
            target_feature_dim = args.c_out
        elif args.features == 'S':
            num_features_out_per_base_model = 1
            target_feature_dim = 1
        else:
            raise ValueError(f"未知的特征类型: {args.features}")

        meta_model_input_dim = len(self.base_models) * num_features_out_per_base_model
        if meta_model_input_dim == 0:
             raise ValueError(f"计算得到的元模型输入维度 meta_model_input_dim 为 0。")

        # 元模型定义 
        args = self.args
        if args.meta_learner_type == 'linear_model':
            self.meta_model = nn.Linear(meta_model_input_dim, target_feature_dim).to(self.device)
            print(f"元模型结构: {self.meta_model}")
        elif args.meta_learner_type == 'weighted_average':
            self.meta_model = nn.Identity()
            print("元学习器类型: 加权平均")
        elif args.meta_learner_type == 'equal_weighted':
            self.meta_model = nn.Identity()
            print("元学习器类型: 均分权重 equal_weighted")
        elif args.meta_learner_type == 'mlp':
            self.meta_model = nn.Sequential(
                nn.Linear(meta_model_input_dim, args.meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.meta_dropout_rate),
                nn.LayerNorm(args.meta_hidden_dim),
                nn.Linear(args.meta_hidden_dim, target_feature_dim)
            ).to(self.device)
            print(f"元模型结构: {self.meta_model}")
        elif args.meta_learner_type == 'gating_network':
            num_base_models = len(self.base_models)
            # c_out: 每个基模型在融合前输出的特征维度
            c_out_per_model = 1 if args.features in ['S', 'MS'] else args.c_out
            gating_input_dim = num_base_models * c_out_per_model # 门控网络的输入维度

            # ==================== 用下面的代码替换你现有的gating_network实例化逻辑 ====================
            self.meta_model = GatedFusionModel(
                num_base_models=num_base_models,
                c_out=c_out_per_model,
                gating_input_dim=gating_input_dim,
                gating_hidden_dim=args.meta_hidden_dim,
                gating_dropout_rate=args.meta_dropout_rate,
                use_context_aware=args.use_context_aware,
                # 传入构建ContextEncoder所需的参数
                seq_len=args.seq_len, 
                enc_in=args.enc_in, 
                context_output_dim=args.context_output_dim 
            ).to(self.device)

            print(f"元模型结构: GatedFusionModel (门控融合网络)")
            print(f" - 门控网络(预测)输入维度: {gating_input_dim}, 隐藏层维度: {args.meta_hidden_dim}")
            if args.use_context_aware:
                print(f" - 上下文感知已开启 (CNN Encoder)，增加特征维度: {args.context_output_dim}")
        # ======================================================================================
        elif args.meta_learner_type == 'attention_fusion':
            # 直接将整个 args 传递给模型，让其自行解析所需参数
            self.meta_model = AttentionFusionModel(self.args).to(self.device)

            print(f"元模型结构: AttentionFusionModel (基于Cross-Attention的门控融合网络)")
            print(f" - 注意力维度 (d_model): {self.args.d_model}, 注意力头数 (n_heads): {self.args.n_heads}")
            print(f" - 历史序列编码器输入维度 (enc_in): {self.args.enc_in}")
        elif args.meta_learner_type == 'context_only_predictor':
            self.meta_model = ContextOnlyPredictor(self.args).to(self.device)
            print(f"元模型结构: ContextOnlyPredictor (仅使用历史上下文的预测器)")
        else:
            raise ValueError(f"未知的元学习器类型: {args.meta_learner_type}.")

        # self.model 将是元模型 (在Exp_Basic中被广泛使用)
        # 注意：在父类 Exp_Basic 的 __init__ 中，self.model = self._build_model()
        # 所以这里返回元模型是正确的。
        model_to_return = self.meta_model 

        if args.use_multi_gpu and args.use_gpu and torch.cuda.device_count() > 1:
            print(f"使用多GPU: {args.device_ids}")
            for i in range(len(self.base_models)):
                self.base_models[i] = nn.DataParallel(self.base_models[i], device_ids=args.device_ids)
            model_to_return = nn.DataParallel(self.meta_model, device_ids=args.device_ids)
            
        return model_to_return # 这个返回值实际上赋值给了 Exp_Basic 中的 self.model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch_base_model(self, base_model, batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim):
        """ 单个基模型处理一个批次数据并返回其原始预测 """
        dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        # 确保模型在正确的设备上
        batch_x = batch_x.to(self.device)
        batch_x_mark = batch_x_mark.to(self.device)
        dec_inp = dec_inp.to(self.device)
        batch_y_mark = batch_y_mark.to(self.device)

        if self.args.output_attention: # 假设 output_attention 适用于基模型
            output = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            output = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # 返回完整的预测，切片在 _process_batch_for_meta_features 中进行
        return output

    def _process_batch_for_meta_features(self, batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim):
        """获取所有基模型的输出并拼接它们作为元特征。在eval模式和no_grad下调用。"""
        base_model_outputs_raw = []
        for model_instance in self.base_models: # self.base_models 存储了已训练和冻结的基模型
            output = self._process_one_batch_base_model(model_instance, batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
            base_model_outputs_raw.append(output)

        base_model_outputs_sliced = []
        for raw_output in base_model_outputs_raw:
            sliced_output = raw_output[:, -self.args.pred_len:, f_dim:]
            if self.args.features == 'MS' and sliced_output.ndim == 2: # 单变量预测时确保有 channel 维度
                sliced_output = sliced_output.unsqueeze(-1)
            base_model_outputs_sliced.append(sliced_output)
        
        if not base_model_outputs_sliced:
            raise RuntimeError("没有基模型输出可以拼接。")

        stacked_features = torch.cat(base_model_outputs_sliced, dim=-1)
        return stacked_features

    def _generate_meta_features(self, data_loader, f_dim, desc=""):
        print(f"开始生成元特征 ({desc})...")
        # 确保所有基模型都处于评估模式
        for bm in self.base_models:
            bm.eval()

        all_stacked_features = []
        all_batch_y_final_target = []
        all_batch_x_original = [] # <--- 新增

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_original = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                stacked_features = self._process_batch_for_meta_features(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                
                batch_y_final_target = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.features == 'MS' and batch_y_final_target.ndim == 2:
                    batch_y_final_target = batch_y_final_target.unsqueeze(-1)

                all_stacked_features.append(stacked_features.detach().cpu())
                all_batch_y_final_target.append(batch_y_final_target.detach().cpu())
                all_batch_x_original.append(batch_x.cpu()) # <--- 新增
        
        print(f"元特征生成完毕 ({desc}).")
        return torch.cat(all_stacked_features, dim=0), torch.cat(all_batch_y_final_target, dim=0), torch.cat(all_batch_x_original, dim=0)

    def _evaluate_model(self, model_to_eval, data_loader, criterion, f_dim, is_meta_model=False, current_base_model_idx=None):
        model_to_eval.eval()
        total_loss = []
        
        with torch.no_grad():
            if is_meta_model: # 评估元模型
                for meta_x_batch, meta_y_batch, meta_x_original_batch in data_loader: # <--- 修改这里
                    meta_x_batch = meta_x_batch.to(self.device)
                    meta_y_batch = meta_y_batch.to(self.device)
                    meta_x_original_batch = meta_x_original_batch.to(self.device) # <--- 新增

                    if self.args.meta_learner_type in ['linear_model', 'mlp']:
                        outputs = self.model(meta_x_batch)
                    
                    elif self.args.meta_learner_type in ['gating_network', 'attention_fusion']:
                        if self.args.meta_learner_type == 'gating_network':
                            context_input = meta_x_original_batch if self.args.use_context_aware else None
                        else: # attention_fusion
                            context_input = meta_x_original_batch
                        outputs, _ = model_to_eval(meta_x_batch, context_input=context_input)
                    
                    else: # Fallback for weighted_average etc., which don't need evaluation here
                        outputs = meta_y_batch # Assign true value to avoid error, loss will be 0

                    loss = criterion(outputs, meta_y_batch)
                    total_loss.append(loss.item())
            else: # 评估基模型
                if current_base_model_idx is None:
                    raise ValueError("评估基模型时需要 current_base_model_idx")
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y_original = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # 重新构建dec_inp
                    dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # 独立的前向传播
                    if self.args.output_attention:
                        outputs = model_to_eval(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model_to_eval(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # 基模型的目标是原始数据的对应部分
                    pred = outputs[:, -self.args.pred_len:, f_dim:]
                    true = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS': # 单变量预测时确保维度一致
                        if pred.ndim == 2: pred = pred.unsqueeze(-1)
                        if true.ndim == 2: true = true.unsqueeze(-1)
                    
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
        
        avg_loss = np.average(total_loss) if total_loss else float('inf')
        model_to_eval.train() # 恢复训练模式，由主循环控制
        return avg_loss
    
    def _calculate_error_correlation(self, vali_loader, criterion, f_dim):
        print("Calculating base model error correlation on validation set...")
        
        # 确保所有基模型都处于评估模式
        for bm in self.base_models:
            bm.eval()

        all_errors = [[] for _ in self.base_models]

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_y_original = batch_y.float().to(self.device)
                true = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)

                for model_idx, base_model in enumerate(self.base_models):
                    # 每个模型独立进行前向传播
                    dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    batch_x_device = batch_x.float().to(self.device)
                    batch_x_mark_device = batch_x_mark.float().to(self.device)
                    batch_y_mark_device = batch_y_mark.float().to(self.device)
                    
                    if self.args.output_attention:
                        pred_raw = base_model(batch_x_device, batch_x_mark_device, dec_inp, batch_y_mark_device)[0]
                    else:
                        pred_raw = base_model(batch_x_device, batch_x_mark_device, dec_inp, batch_y_mark_device)
                    
                    pred = pred_raw[:, -self.args.pred_len:, f_dim:]
                    
                    # 确保维度一致
                    if self.args.features == 'MS':
                        if pred.ndim == 2: pred = pred.unsqueeze(-1)
                        if true.ndim == 2: true = true.unsqueeze(-1)
                    
                    # 计算误差并保存
                    error = true - pred
                    all_errors[model_idx].append(error.cpu().numpy())

        # 将所有批次的误差拼接起来
        for i in range(len(all_errors)):
            all_errors[i] = np.concatenate(all_errors[i], axis=0)
        
        # 将误差展平为一维向量以计算相关性
        num_models = len(self.base_models)
        errors_flat = [err.flatten() for err in all_errors]
        
        # 计算两两之间的皮尔逊相关系数
        correlation_results = {}
        model_names = self.args.base_model_names
        for i in range(num_models):
            for j in range(i + 1, num_models):
                corr, _ = pearsonr(errors_flat[i], errors_flat[j])
                key = f"Corr({model_names[i]}_vs_{model_names[j]})"
                correlation_results[key] = corr
        
        print(f"Error Correlation Results: {correlation_results}")
        return correlation_results

    # ==================== 用下面两个新方法替换旧的 _plot_case_study ====================
    def _plot_case_study_predictions(self, save_path, true_data, final_pred, base_model_preds, base_model_names, title):
        """
        [实验一-子任务] 绘制单个特征的预测对比图。
        """
        plt.figure() # 移除固定的 figsize，使用默认大小
        
        # 绘制真实值和最终融合预测
        plt.plot(true_data, label='GroundTruth', color='black', linewidth=2.5)
        plt.plot(final_pred, label='Fused Prediction', color='red', linestyle='--', linewidth=2)
        
        # 绘制所有基模型的预测
        num_base_models = len(base_model_preds)
        # 使用 tab10 配色方案以获得更好的对比度
        colors = plt.cm.get_cmap('tab10', num_base_models)
        for i in range(num_base_models):
            # 移除 alpha 透明度，设置与 Fused Prediction 一致的 linewidth
            plt.plot(base_model_preds[i], label=f'{base_model_names[i]} Pred', color=colors(i), linestyle=':', linewidth=2)
            
        plt.title(title, fontsize=16)
        plt.xlabel('Time Step in Prediction Horizon')
        plt.ylabel('Value')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--') # 移除 alpha 透明度
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_case_study_weights(self, save_path, dynamic_weights, pred_len, base_model_names, title):
        """
        [实验一-子任务] 绘制动态权重变化图。
        """
        plt.figure() # 移除固定的 figsize，使用默认大小
        num_base_models = dynamic_weights.shape[1]
        # 使用 tab10 配色方案以获得更好的对比度
        colors = plt.cm.get_cmap('tab10', num_base_models)

        for i in range(num_base_models):
            plt.plot(range(pred_len), dynamic_weights[:, i], label=f'{base_model_names[i]} Weight', color=colors(i), marker='o', linestyle='--')

        plt.title(title, fontsize=16)
        plt.xlabel('Time Step in Prediction Horizon')
        plt.ylabel('Assigned Weight')
        plt.ylim(-0.05, 1.05)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--') # 移除 alpha 透明度
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    # =================================================================================

    # ==================== 用下面的优化版本替换现有的 _plot_all_predictions ====================
    def _plot_all_predictions(self, save_path, true_data, final_pred, all_base_preds, base_model_names, title):
        """
        绘制所有基模型和最终融合预测与真实值的对比图 (优化后版本)。
        """
        plt.figure() # 移除固定的 figsize，使用默认大小
        
        # 1. 绘制真实值和最终融合预测 (使用更粗的线条)
        plt.plot(true_data.flatten(), label='Ground Truth', color='black', linewidth=3, zorder=5)
        plt.plot(final_pred.flatten(), label='Fused Prediction', color='red', linestyle='--', linewidth=2.5, zorder=4)

        # 2. 绘制所有基模型的预测
        num_base_models = len(all_base_preds)
        colors = plt.cm.get_cmap('tab10', num_base_models)
        
        for i, base_pred in enumerate(all_base_preds):
            plt.plot(
                base_pred.flatten(), 
                label=f'{base_model_names[i]}', 
                color=colors(i), 
                linestyle='-', 
                linewidth=1.5
                # 移除了 alpha 透明度控制
            )
            
        plt.title(title, fontsize=12) 
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        
        # 3. 将图例(legend)的字体调大
        plt.legend(fontsize='large')
        
        plt.grid(True, linestyle='--') # 移除了 alpha 透明度控制
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    # =======================================================================================

    def train(self, setting):
        # --- 基本设置 ---
        train_data_raw, train_loader_raw = self._get_data(flag='train')
        vali_data_raw, vali_loader_raw = self._get_data(flag='val')
        # test_data_raw, test_loader_raw = self._get_data(flag='test') # 测试集通常不在训练阶段使用进行早停

        main_checkpoint_path_dir = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(main_checkpoint_path_dir):
            os.makedirs(main_checkpoint_path_dir)

        criterion = self._select_criterion()
        f_dim = -1 if self.args.features == 'MS' else 0
        
        # --- 阶段一: 训练基模型 ---
        print("\n Phase 1: Train Base models...")
        base_model_val_losses = [] # <--- 新增：用于存储验证损失的列表
        for i, base_model in enumerate(self.base_models):
            print(f"\nTrain base model {i+1}/{len(self.base_models)}: {self.args.base_model_names[i]}")
            
            # 为每个基模型实例创建独立的优化器和 EarlyStopping
            # 如果模型是 DataParallel 包装的，参数在 .module 中
            model_params = base_model.module.parameters() if isinstance(base_model, nn.DataParallel) else base_model.parameters()
            bm_optim = optim.Adam(model_params, lr=self.args.learning_rate)
            
            # EarlyStopping 的 path 参数是目录，它会在该目录下创建 checkpoint.pth
            # 为了给每个基模型保存独立的检查点，我们需要让它们使用不同的目录，或者在之后重命名/复制
            # 这里选择每个基模型训练完成后，将它的最佳状态从通用 EarlyStopping 路径复制到特定路径
            current_model_checkpoint_dir = os.path.join(main_checkpoint_path_dir, f"temp_bm_{i}")
            os.makedirs(current_model_checkpoint_dir, exist_ok=True)
            bm_early_stopping = EarlyStopping(patience=self.args.base_model_patience, verbose=True)

            if self.args.use_amp:
                bm_scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.base_model_train_epochs):
                iter_count = 0
                train_loss_list = []
                base_model.train()
                epoch_time = time.time()

                for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader_raw):
                    iter_count += 1
                    bm_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y_original = batch_y.float().to(self.device) # 全量 batch_y
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # 基模型的目标
                    true_y_for_loss = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.features == 'MS' and true_y_for_loss.ndim == 2:
                        true_y_for_loss = true_y_for_loss.unsqueeze(-1)

                    # 重构这部分代码以避免计算图重复使用问题
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            # 重新构建dec_inp以确保独立的计算图
                            dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                            
                            # 独立的前向传播
                            if self.args.output_attention:
                                pred_y_bm = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                pred_y_bm = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                
                            pred_y_for_loss = pred_y_bm[:, -self.args.pred_len:, f_dim:]
                            if self.args.features == 'MS' and pred_y_for_loss.ndim == 2:
                                pred_y_for_loss = pred_y_for_loss.unsqueeze(-1)
                            loss = criterion(pred_y_for_loss, true_y_for_loss)
                        bm_scaler.scale(loss).backward()
                        bm_scaler.step(bm_optim)
                        bm_scaler.update()
                    else:
                        # 重新构建dec_inp以确保独立的计算图
                        dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        # 独立的前向传播
                        if self.args.output_attention:
                            pred_y_bm = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            pred_y_bm = base_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                        pred_y_for_loss = pred_y_bm[:, -self.args.pred_len:, f_dim:]
                        if self.args.features == 'MS' and pred_y_for_loss.ndim == 2:
                            pred_y_for_loss = pred_y_for_loss.unsqueeze(-1)
                        loss = criterion(pred_y_for_loss, true_y_for_loss)
                        loss.backward()
                        bm_optim.step()
                    
                    train_loss_list.append(loss.item())
                
                epoch_train_loss_avg = np.average(train_loss_list) if train_loss_list else float('inf')
                # print(f"Modelo Base {i+1} - Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")
                
                bm_vali_loss = self._evaluate_model(base_model, vali_loader_raw, criterion, f_dim, is_meta_model=False, current_base_model_idx=i)
                
                print(f"Base model {i+1} Epoch: {epoch + 1} | Train Loss: {epoch_train_loss_avg:.7f} Vali Loss: {bm_vali_loss:.7f}")
                
                bm_early_stopping(bm_vali_loss, base_model, path=current_model_checkpoint_dir) # EarlyStopping 内部会加载最佳模型
                if bm_early_stopping.early_stop:
                    print(f"Modelo Base {i+1} - Early stopping na epoch {epoch+1}")
                    break
                adjust_learning_rate(bm_optim, epoch + 1, self.args) # 假设学习率调整策略对基模型也适用

            # 保存最佳基模型状态 (EarlyStopping 已经将最佳权重加载到 base_model)
            best_bm_path = os.path.join(main_checkpoint_path_dir, f'base_model_{i}_checkpoint.pth')
            
            # 处理 DataParallel
            state_dict_to_save = base_model.module.state_dict() if isinstance(base_model, nn.DataParallel) else base_model.state_dict()
            torch.save(state_dict_to_save, best_bm_path)
            print(f"base model {i+1} save path {best_bm_path}")

            # 2. <--- 新增：记录该模型的最佳验证损失
            base_model_val_losses.append(bm_early_stopping.val_loss_min)
            print(f"基模型 {i+1} 的最佳验证损失为: {bm_early_stopping.val_loss_min:.7f}")

            # 冻结参数并设置为评估模式
            base_model.eval()
            for param in base_model.parameters():
                param.requires_grad = False
            print(f"Frozen base model {i+1} parameters.")

        if len(self.base_models) > 1 and self.args.meta_learner_type == 'attention_fusion' and self.args.calculate_correlation:
            print("Calculating error correlation...")

            # 调用新函数计算误差相关性
            correlation_results = self._calculate_error_correlation(vali_loader_raw, criterion, f_dim)
            
            # 准备要记录的信息
            model_correlation_workresult_path = f"workresult/stacking/correlation/{self.args.model}_error_correlation.csv" 
            record_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_name': self.args.data_name,
                **correlation_results
            }
            
            # 将结果追加到CSV文件中
            df_entry = pd.DataFrame([record_entry])
            if not os.path.exists(model_correlation_workresult_path):
                df_entry.to_csv(model_correlation_workresult_path, mode='w', header=True, index=False)
            else:
                df_entry.to_csv(model_correlation_workresult_path, mode='a', header=False, index=False)
            print(f"Error correlation results saved to {model_correlation_workresult_path}")

        if self.args.meta_learner_type in ['linear_model', 'mlp', 'gating_network', 'attention_fusion']: 
            # --- 阶段二: 生成元特征 ---
            print("\n Phase 2: Generate Meta-Features...")
            # <--- 修改这里，接收第三个返回值
            X_meta_train, y_meta_train, X_original_train = self._generate_meta_features(train_loader_raw, f_dim, desc="treino")
            X_meta_val, y_meta_val, X_original_val = self._generate_meta_features(vali_loader_raw, f_dim, desc="validação")

            # <--- 修改这里，创建包含三个部分的数据集
            meta_train_dataset = TensorDataset(X_meta_train, y_meta_train, X_original_train)
            meta_val_dataset = TensorDataset(X_meta_val, y_meta_val, X_original_val)

            
            # 使用与原始数据加载器相似的批量大小，或者可以为其定义新的批量大小
            # self.args.batch_size 可能不适合元特征，因为元特征数量可能较少
            # 但为了简单起见，先用它
            meta_batch_size = self.args.batch_size 
            meta_train_loader = DataLoader(meta_train_dataset, batch_size=meta_batch_size, shuffle=True)
            meta_val_loader = DataLoader(meta_val_dataset, batch_size=meta_batch_size, shuffle=False)

            # --- 阶段三: 训练元模型 ---
            # self.model 在 _build_model 中被赋值为 self.meta_model
            print("\n Phase 3: Train Meta-Model...")
            
            meta_model_params = self.model.module.parameters() if isinstance(self.model, nn.DataParallel) else self.model.parameters()
            meta_optim = optim.Adam(meta_model_params, lr=self.args.learning_rate) # 可以为元模型设置不同学习率
            
            # 元模型的 EarlyStopping 保存到主 checkpoint 路径，与 test 方法兼容
            meta_early_stopping_dir = main_checkpoint_path_dir # EarlyStopping 将在此目录下创建 checkpoint.pth
            meta_early_stopping = EarlyStopping(patience=self.args.meta_model_patience, verbose=True)

            if self.args.use_amp:
                meta_scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.meta_model_train_epochs):
                meta_train_loss_list = []
                self.model.train() # 元模型训练模式
                epoch_time = time.time()

                for meta_x_batch, meta_y_batch, meta_x_original_batch in meta_train_loader: # <--- 修改这里
                    meta_optim.zero_grad()
                    meta_x_batch = meta_x_batch.to(self.device)
                    meta_y_batch = meta_y_batch.to(self.device)
                    meta_x_original_batch = meta_x_original_batch.to(self.device) # <--- 新增

                    # ==================== 从这里开始是核心修改 ====================
                    # 将 forward, loss, backward, step 合并在一个逻辑块内
                    if self.args.meta_learner_type in ['linear_model', 'mlp']:
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                meta_outputs = self.model(meta_x_batch)
                                loss = criterion(meta_outputs, meta_y_batch)
                            meta_scaler.scale(loss).backward()
                            meta_scaler.step(meta_optim)
                            meta_scaler.update()
                        else:
                            meta_outputs = self.model(meta_x_batch)
                            loss = criterion(meta_outputs, meta_y_batch)
                            loss.backward()
                            meta_optim.step()
                    elif self.args.meta_learner_type == 'context_only_predictor':
                        # 这个模型只使用 context_input
                        context_input = meta_x_original_batch
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                # 它只返回一个输出
                                meta_outputs, _ = self.model(None, context_input=context_input)
                                loss = criterion(meta_outputs, meta_y_batch)
                            meta_scaler.scale(loss).backward()
                            meta_scaler.step(meta_optim)
                            meta_scaler.update()
                        else:
                            meta_outputs, _ = self.model(None, context_input=context_input)
                            loss = criterion(meta_outputs, meta_y_batch)
                            loss.backward()
                            meta_optim.step()
                    elif self.args.meta_learner_type in ['gating_network', 'attention_fusion']:
                        if self.args.meta_learner_type == 'gating_network':
                            context_input = meta_x_original_batch if self.args.use_context_aware else None
                        else:  # attention_fusion
                            context_input = meta_x_original_batch

                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                meta_outputs, _ = self.model(meta_x_batch, context_input=context_input)
                                loss = criterion(meta_outputs, meta_y_batch)
                            meta_scaler.scale(loss).backward()
                            meta_scaler.step(meta_optim)
                            meta_scaler.update()
                        else:
                            meta_outputs, _ = self.model(meta_x_batch, context_input=context_input)
                            loss = criterion(meta_outputs, meta_y_batch)
                            loss.backward()
                            meta_optim.step()
                    
                    meta_train_loss_list.append(loss.item())
                
                epoch_meta_train_loss_avg = np.average(meta_train_loss_list) if meta_train_loss_list else float('inf')
                # print(f"Meta-Modelo - Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")

                meta_vali_loss = self._evaluate_model(self.model, meta_val_loader, criterion, f_dim, is_meta_model=True)
                
                print(f"Meta-Model Epoch: {epoch + 1} | Treino Loss: {epoch_meta_train_loss_avg:.7f} Vali Loss: {meta_vali_loss:.7f}")
                
                meta_early_stopping(meta_vali_loss, self.model, path=meta_early_stopping_dir) # EarlyStopping 会加载最佳元模型
                if meta_early_stopping.early_stop:
                    print(f"Meta-Modelo - Early stopping na epoch {epoch+1}")
                    break
                adjust_learning_rate(meta_optim, epoch + 1, self.args) # 调整元模型学习率
            
        elif self.args.meta_learner_type == 'weighted_average':
            print("\n阶段二/三: 计算元学习器权重...")
            losses_tensor = torch.tensor(base_model_val_losses, dtype=torch.float32)
            
            # 使用 softmax 计算权重
            tau = self.args.softmax_temperature
            self.meta_weights = F.softmax(-losses_tensor / tau, dim=0)

            print("基模型验证损失:", [f"{loss:.4f}" for loss in base_model_val_losses])
            print(f"计算出的权重 (温度={tau}):", [f"{w:.4f}" for w in self.meta_weights.tolist()])
            
            # 将权重保存到磁盘，以便test时加载
            weights_path = os.path.join(main_checkpoint_path_dir, 'meta_weights.pth')
            torch.save(self.meta_weights, weights_path)
            print(f"元学习器权重已计算并保存。")

        elif self.args.meta_learner_type == 'equal_weighted':
            print("\n阶段二/三: 计算元学习器权重 (均分)...")
            num_base_models = len(self.base_models)
            if num_base_models == 0:
                raise ValueError("没有基模型，无法计算均分权重。")
            
            # 自适应地创建均分权重张量
            equal_weight = 1.0 / num_base_models
            self.meta_weights = torch.full((num_base_models,), equal_weight, dtype=torch.float32)

            print(f"基模型数量: {num_base_models}")
            print(f"计算出的均分权重:", [f"{w:.4f}" for w in self.meta_weights.tolist()])
            
            # 将权重保存到磁盘，路径和文件名与 'weighted_average' 保持一致
            # 这样 test 方法就可以复用加载逻辑
            weights_path = os.path.join(main_checkpoint_path_dir, 'meta_weights.pth')
            torch.save(self.meta_weights, weights_path)
            print(f"元学习器权重已计算并保存。")

        print("Train model over.")
        return self.model # 返回训练好的元模型
        


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        main_checkpoint_path_dir = os.path.join(self.args.checkpoints, setting)

        # ==================== 在这里增加以下代码 ====================
        # 实验分析初始化
        analysis_folder_path = os.path.join('./workresult/stacking')
        if self.args.do_exp1_case_study or self.args.do_exp2_weight_dist:
            os.makedirs(analysis_folder_path, exist_ok=True)

        # 实验一：案例分析所需参数
        samples_to_visualize = []
        if self.args.do_exp1_case_study:
            try:
                samples_to_visualize = [int(i.strip()) for i in self.args.exp1_samples_to_visualize.split(',')]
                print(f"实验一: 将对样本索引 {samples_to_visualize} 进行案例分析。")
            except:
                print(f"警告: 无法解析 'exp1_samples_to_visualize' 参数: {self.args.exp1_samples_to_visualize}。实验一将不会执行。")
                self.args.do_exp1_case_study = False

        # 实验二：权重分布所需列表
        all_dynamic_weights = [] if self.args.do_exp2_weight_dist else None
        # =========================================================

        if test: # test=1 表示从磁盘加载模型
            print('Loading models for testing. It...')

            if self.args.meta_learner_type in ['weighted_average', 'equal_weighted']: 
                # 加载 meta_weights.pth
                weights_path = os.path.join(main_checkpoint_path_dir, 'meta_weights.pth')
                if not os.path.exists(weights_path): raise FileNotFoundError(f"元权重文件未找到: {weights_path}")
                self.meta_weights = torch.load(weights_path, map_location=self.device)
            elif self.args.meta_learner_type in ['linear_model', 'mlp', 'gating_network', 'attention_fusion', 'context_only_predictor']:
                # 1. 加载元模型 (self.model)
                meta_model_checkpoint_file = os.path.join(main_checkpoint_path_dir, 'checkpoint.pth')
                if not os.path.exists(meta_model_checkpoint_file):
                    # Fallback (尝试匹配旧代码中可能的路径结构)
                    alt_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                    if os.path.exists(alt_path): meta_model_checkpoint_file = alt_path
                    else: raise FileNotFoundError(f"Checkpoint do meta-modelo não encontrado: {meta_model_checkpoint_file}")
                
                print(f"Loading meta-model checkpoint from: {meta_model_checkpoint_file}")
                # 处理 DataParallel 包装的模型
                model_to_load_meta = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_to_load_meta.load_state_dict(torch.load(meta_model_checkpoint_file, map_location=self.device))
            else:
                raise ValueError(f"Invalid meta-learner type: {self.args.meta_learner_type}")

            # 2. 加载所有基模型 (self.base_models)
            for i, bm in enumerate(self.base_models):
                base_model_checkpoint_file = os.path.join(main_checkpoint_path_dir, f'base_model_{i}_checkpoint.pth')
                if not os.path.exists(base_model_checkpoint_file):
                    alt_path = os.path.join('./checkpoints/' + setting, f'base_model_{i}_checkpoint.pth')
                    if os.path.exists(alt_path): base_model_checkpoint_file = alt_path
                    else: raise FileNotFoundError(f"Checkpoint do modelo base {i} não encontrado: {base_model_checkpoint_file}")

                print(f"Loading base model checkpoint {i} from: {base_model_checkpoint_file}")
                model_to_load_base = bm.module if isinstance(bm, nn.DataParallel) else bm
                model_to_load_base.load_state_dict(torch.load(base_model_checkpoint_file, map_location=self.device))
                bm.eval() # 确保是评估模式
                for param in bm.parameters(): # 确保参数冻结
                    param.requires_grad = False

        # --- 开始测试 ---
        preds_for_metric, trues_for_metric = [], []
        preds_inv, trues_inv = [], [] 

        results_folder_path = './results/' + setting + '/'
        if not os.path.exists(results_folder_path): os.makedirs(results_folder_path)
        visualization_folder_path = f'./workresult/stacking/visual/{self.args.meta_learner_type}/{self.args.data_name}'
        if not os.path.exists(visualization_folder_path): os.makedirs(visualization_folder_path)

        # 设置所有模型为评估模式 (如果 train 后直接 test，它们应该已经是 eval 状态，但再次设置无害)
        for bm in self.base_models:
            bm.eval()
        self.model.eval() # 元模型

        # ==================== 高效诊断逻辑：初始化 ====================
        # 创建一个列表的列表，用于存储每个基模型的独立预测结果
        # 例如 all_preds_bm[0] 将存储基模型0的所有批次预测
        all_preds_bm = [[] for _ in self.base_models]
        all_trues_for_diag = [] # 真实值对所有模型都是一样的，存一份即可
        # ==========================================================

        f_dim = -1 if self.args.features == 'MS' else 0
        # 在 with torch.no_grad() 前面定义 c_out
        c_out = 1 if self.args.features in ['S', 'MS'] else self.args.c_out

        with torch.no_grad():
            for i, (batch_x, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_original = batch_y_original.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 1. 使用基模型生成元特征
                stacked_features = self._process_batch_for_meta_features(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                
                # 2. <--- 修改：根据元学习器类型计算最终输出
                final_outputs = None
                dynamic_weights = None # 初始化

                if self.args.meta_learner_type in ['linear_model', 'mlp']: 
                    final_outputs = self.model(stacked_features)
                elif self.args.meta_learner_type == 'context_only_predictor':
                    # 这个模型只使用 batch_x (历史序列)
                    final_outputs, _ = self.model(None, context_input=batch_x)
                elif self.args.meta_learner_type in ['gating_network', 'attention_fusion']:
                    context_input = batch_x if (self.args.use_context_aware or self.args.meta_learner_type == 'attention_fusion') else None
                    # 注意：attention_fusion 模型 forward 需要两个参数 (stacked_features, context_input)
                    final_outputs, dynamic_weights = self.model(stacked_features, context_input=context_input)
                elif self.args.meta_learner_type in ['weighted_average', 'equal_weighted']: 
                    # 对 stacked_features 进行加权求和
                    bs, pred_len, _ = stacked_features.shape
                    c_out = 1 if self.args.features == 'S' or self.args.features == 'MS' else self.args.c_out
                    reshaped_features = stacked_features.view(bs, pred_len, len(self.base_models), c_out)
                    weights_view = self.meta_weights.to(stacked_features.device).view(1, 1, -1, 1)
                    final_outputs = (reshaped_features * weights_view).sum(dim=2)

                # 准备真实标签
                target_y = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.features == 'MS' and target_y.ndim == 2:
                    target_y = target_y.unsqueeze(-1)

                numpy_outputs_scaled = final_outputs.detach().cpu().numpy()
                numpy_target_scaled = target_y.detach().cpu().numpy()

                preds_for_metric.append(numpy_outputs_scaled)
                trues_for_metric.append(numpy_target_scaled)

                for k in range(len(self.base_models)):
                    # 切片操作，提取第 k 个模型的预测
                    pred_k = stacked_features[:, :, k * c_out:(k + 1) * c_out]
                    all_preds_bm[k].append(pred_k.detach().cpu().numpy())

                # 真实值只需在每个批次存一次
                all_trues_for_diag.append(target_y.detach().cpu().numpy())
                
                current_preds_to_save = numpy_outputs_scaled
                current_trues_to_save = numpy_target_scaled

                # 逆变换逻辑 (与之前相同)
                if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
                    try:
                        shape_out = numpy_outputs_scaled.shape
                        # 确保逆变换作用于正确的维度，通常是最后一个特征维度
                        current_preds_to_save = test_data.inverse_transform(numpy_outputs_scaled.reshape(-1, shape_out[-1])).reshape(shape_out)
                        shape_true = numpy_target_scaled.shape
                        current_trues_to_save = test_data.inverse_transform(numpy_target_scaled.reshape(-1, shape_true[-1])).reshape(shape_true)
                    except Exception as e:
                        print(f"Aviso: Falha na transformação inversa durante o teste: {e}")
                        # Fallback: use scaled data if inverse transform fails
                        current_preds_to_save = numpy_outputs_scaled
                        current_trues_to_save = numpy_target_scaled

                preds_inv.append(current_preds_to_save)
                trues_inv.append(current_trues_to_save)

                # ==================== 在这里增加以下代码 ====================
                # 实验分析逻辑 (在循环内)
                # 仅当元模型是门控网络时执行
                if self.args.meta_learner_type in ['gating_network', 'attention_fusion'] and dynamic_weights is not None:

                    # 实验一: 案例分析
                    if self.args.do_exp1_case_study:
                        batch_size = batch_x.shape[0]
                        global_sample_idx_start = i * batch_size
                        for sample_idx in samples_to_visualize:
                            if global_sample_idx_start <= sample_idx < global_sample_idx_start + batch_size:
                                print(f"正在为样本 {sample_idx} 生成案例分析图...")
                                local_idx = sample_idx - global_sample_idx_start
                                
                                # --- 准备数据 (保持多维) ---
                                # 提取该样本所需的所有数据，不使用.flatten()
                                true_sample = target_y[local_idx].detach().cpu().numpy()
                                final_pred_sample = final_outputs[local_idx].detach().cpu().numpy()
                                weights_sample = dynamic_weights[local_idx].detach().cpu().numpy()
                                
                                # 提取基模型预测 (保持多维)
                                base_preds_sample = []
                                for k in range(len(self.base_models)):
                                    pred_k = stacked_features[local_idx, :, k * c_out:(k + 1) * c_out]
                                    base_preds_sample.append(pred_k.detach().cpu().numpy())
                                
                                # --- 绘图逻辑 ---
                                case_study_path = os.path.join(analysis_folder_path, 'exp1/', self.args.meta_learner_type, self.args.data_name, f'sample_{sample_idx}')
                                os.makedirs(case_study_path, exist_ok=True)
                                num_features = true_sample.shape[-1]

                                # 1. 绘制权重图 (每个样本只需一张)
                                weights_save_file = os.path.join(case_study_path, f'weights_analysis.pdf')
                                self._plot_case_study_weights(
                                    save_path=weights_save_file,
                                    dynamic_weights=weights_sample,
                                    pred_len=self.args.pred_len,
                                    base_model_names=self.args.base_model_names,
                                    title=f'Weight Dynamics for Sample {sample_idx}'
                                )

                                # 2. 循环遍历每个特征，为每个特征绘制一张预测对比图
                                for feature_idx in range(num_features):
                                    pred_save_file = os.path.join(case_study_path, f'feature_{feature_idx}_prediction_comparison.pdf')
                                    
                                    # 提取当前特征的基模型预测
                                    base_preds_for_feature = [pred[:, feature_idx] for pred in base_preds_sample]

                                    self._plot_case_study_predictions(
                                        save_path=pred_save_file,
                                        true_data=true_sample[:, feature_idx],
                                        final_pred=final_pred_sample[:, feature_idx],
                                        base_model_preds=base_preds_for_feature,
                                        base_model_names=self.args.base_model_names,
                                        title=f'Prediction Comparison for Sample {sample_idx}, Feature {feature_idx}'
                                    )

                    # 实验二: 收集权重
                    if self.args.do_exp2_weight_dist:
                        all_dynamic_weights.append(dynamic_weights.detach().cpu().numpy())
                # =========================================================

                if i % 20 == 0: # 添加一个参数控制可视化
                    input_viz = batch_x.detach().cpu().numpy()
                    if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
                        try:
                            shape_in_viz = input_viz.shape
                            input_viz = test_data.inverse_transform(input_viz.reshape(-1, shape_in_viz[-1])).reshape(shape_in_viz)
                        except Exception as e:
                            print(f"Aviso: Falha na transformação inversa do input_viz para visualização: {e}")
                    
                    num_features_to_plot = current_trues_to_save.shape[-1]
                    for feature_idx in range(min(num_features_to_plot, 5)): # 最多可视化3个特征
                        input_hist_for_plot = input_viz[0, :, feature_idx if input_viz.shape[-1] > feature_idx else 0]
                        gt_viz = np.concatenate((input_hist_for_plot, current_trues_to_save[0, :, feature_idx]), axis=0)
                        pd_viz = np.concatenate((input_hist_for_plot, current_preds_to_save[0, :, feature_idx]), axis=0)
                        visual(gt_viz, pd_viz, os.path.join(visualization_folder_path, f"batch{i}_sample0_feature{feature_idx}.pdf"))
                
                # 我们对第一个样本进行可视化，确保图表不会过于拥挤。
                # 替换原来的判断代码
                if current_preds_to_save.shape[0] > 0 and current_preds_to_save.shape[1] > 0:
                    sample_idx = 0  # 选取第一个样本进行可视化
                    num_features = current_preds_to_save.shape[-1]
                    
                    for feature_idx in range(num_features):
                        print(f"正在为特征 {feature_idx} 生成所有模型预测对比图...")

                        # 准备数据
                        true_sample_data = current_trues_to_save[sample_idx, :, feature_idx]
                        final_pred_sample = current_preds_to_save[sample_idx, :, feature_idx]
                        
                        # 准备基模型预测数据 (修正了原先的可视化逻辑)
                        # 1. 从当前批次中提取出第一个样本的预测结果 (scaled)
                        base_preds_sample_scaled = []
                        for k in range(len(self.base_models)):
                            # all_preds_bm[k][-1] 是模型k在当前批次的预测结果
                            pred_k_current_batch = all_preds_bm[k][-1]
                            if pred_k_current_batch.shape[0] > sample_idx:
                                # 提取第一个样本，保持所有特征维度 [pred_len, c_out]
                                base_preds_sample_scaled.append(pred_k_current_batch[sample_idx, :, :])

                        # 2. 如果需要，对提取出的基模型预测进行反归一化
                        base_preds_to_plot_unscaled = []
                        if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
                            try:
                                for pred_k_sample in base_preds_sample_scaled:
                                    # inverse_transform 要求输入形状为 (n_samples, n_features)
                                    # 此处 pred_k_sample 形状为 (pred_len, c_out)，符合要求
                                    pred_k_unscaled = test_data.inverse_transform(pred_k_sample)
                                    base_preds_to_plot_unscaled.append(pred_k_unscaled)
                            except Exception as e:
                                print(f"警告: 为绘图反归一化基模型预测失败: {e}。将使用归一化数据绘图。")
                                # 如果失败，则回退到使用 scaled 数据
                                base_preds_to_plot_unscaled = base_preds_sample_scaled
                        else:
                            # 如果不进行反归一化，直接使用 scaled 数据
                            base_preds_to_plot_unscaled = base_preds_sample_scaled

                        # 3. 从可能已反归一化的数据中，提取当前特征进行绘图
                        base_preds_for_plot = []
                        if base_preds_to_plot_unscaled:
                            for pred_k in base_preds_to_plot_unscaled:
                                if pred_k.shape[-1] > feature_idx:
                                    base_preds_for_plot.append(pred_k[:, feature_idx])

                        # 定义保存路径
                        all_preds_plot_path = os.path.join(visualization_folder_path, f'all_models_comparison_feature_{feature_idx}.pdf')

                        # 确保文件路径的目录存在
                        plot_dir = os.path.dirname(all_preds_plot_path)
                        if plot_dir and not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)

                        # 调用绘图函数
                        self._plot_all_predictions(
                            save_path=all_preds_plot_path,
                            true_data=true_sample_data,
                            final_pred=final_pred_sample,
                            all_base_preds=base_preds_for_plot,
                            base_model_names=self.args.base_model_names,
                            title=f'Comparison of Base Models and Fused Prediction (Sample {sample_idx}, Feature {feature_idx})'
                        )
                    
                    print(f"所有模型预测对比图已为所有特征生成。")
                # --- 新增结束 ---


        preds_for_metric = np.concatenate(preds_for_metric, axis=0)
        trues_for_metric = np.concatenate(trues_for_metric, axis=0)
        preds_inv = np.concatenate(preds_inv, axis=0)
        trues_inv = np.concatenate(trues_inv, axis=0)

        # 1. 计算并打印独立基模型的指标
        print("\n--- 独立基模型性能诊断结果 ---")
        all_trues_for_diag = np.concatenate(all_trues_for_diag, axis=0)
        for k in range(len(self.base_models)):
            preds_k = np.concatenate(all_preds_bm[k], axis=0)
            mae_k, mse_k, _, _, _ = metric(preds_k, all_trues_for_diag)
            print(f"基模型 {k} ({self.args.base_model_names[k]}) -> MAE: {mae_k:.4f}, MSE: {mse_k:.4f}")

        print('Test data forms (after concatenation):')
        print(f'preds_for_metric shape: {preds_for_metric.shape} | trues_for_metric shape: {trues_for_metric.shape}')
        

        mae, mse, rmse, mape, mspe = metric(preds_for_metric, trues_for_metric)
        print(f'Metrics in Scaled Data: MSE:{mse:.4f}, MAE:{mae:.4f}')

        if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
            final_preds_to_save = preds_inv
            final_trues_to_save = trues_inv
        else: # 没有逆变换，使用原始缩放数据指标
            final_preds_to_save = preds_for_metric
            final_trues_to_save = trues_for_metric

        final_mae, final_mse, final_rmse, final_mape, final_mspe = mae, mse, rmse, mape, mspe

        dtw_val = -999 # 初始化
        if self.args.use_dtw:
            try:
                from utils.dtw_metric import accelerated_dtw 
                dtw_list = []
                # 对每个样本和每个特征计算DTW，然后平均 (或者只对第一个特征)
                # 为简单起见，这里假设对每个样本的第一个特征计算 DTW
                num_samples_for_dtw = final_preds_to_save.shape[0]
                feature_idx_for_dtw = 0 # 可以选择其他特征或所有特征
                
                for k in range(num_samples_for_dtw):
                    x_dtw = final_preds_to_save[k, :, feature_idx_for_dtw].reshape(-1, 1)
                    y_dtw = final_trues_to_save[k, :, feature_idx_for_dtw].reshape(-1, 1)
                    manhattan_distance = lambda x, y: np.abs(x - y)
                    d, _, _, _ = accelerated_dtw(x_dtw, y_dtw, dist=manhattan_distance)
                    dtw_list.append(d)
                if dtw_list: dtw_val = np.array(dtw_list).mean()
                print(f'DTW (no recurso {feature_idx_for_dtw}): {dtw_val:.4f}')
            except ImportError: print("Aviso: `accelerated_dtw` não encontrado. Pulando DTW.")
            except Exception as e: print(f"Erro durante DTW: {e}. Pulando.")
        
        # 保存结果到 CSV
        result_file_csv = f'{self.args.result_path}.csv'
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'setting': setting, # 添加setting名称
            'dataset': self.args.data_path.split('.')[0], # 通常是文件名不含扩展名
            'base_model_names': str(self.args.base_model_names),
            'meta_hidden_dim': self.args.meta_hidden_dim, # 对于线性模型不那么重要
            'meta_dropout_rate': self.args.meta_dropout_rate,
            'meta_learner_type': self.args.meta_learner_type,
            'softmax_temperature': self.args.softmax_temperature,
            'mae': final_mae, 'mse': final_mse, 'rmse': final_rmse,
            'mape': final_mape, 'mspe': final_mspe, 'dtw': dtw_val

        }
        df_record = pd.DataFrame([record])
        mode_csv = 'w' if not os.path.exists(result_file_csv) else 'a'
        df_record.to_csv(result_file_csv, mode=mode_csv, header=not os.path.exists(result_file_csv) or mode_csv=='w', index=False)
        print(f"Results saved in {result_file_csv}")

        # 保存到 TXT 文件 (与之前格式一致)
        with open("result_long_term_forecast.txt", 'a') as f_res:
            f_res.write(f"{setting}\n")
            f_res.write(f'mse:{final_mse:.4f}, mae:{final_mae:.4f}, rmse:{final_rmse:.4f}, mape:{final_mape:.4f}, mspe:{final_mspe:.4f}, dtw:{dtw_val:.4f}\n\n')

        np.save(os.path.join(results_folder_path, 'metrics.npy'), np.array([final_mae, final_mse, final_rmse, final_mape, final_mspe, dtw_val]))
        np.save(os.path.join(results_folder_path, 'pred.npy'), final_preds_to_save) # 保存最终用于评估的预测
        np.save(os.path.join(results_folder_path, 'true.npy'), final_trues_to_save) # 保存最终用于评估的真实值
        
        # 如果需要，也可以保存缩放版本的数据
        if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
            np.save(os.path.join(results_folder_path, 'pred_scaled.npy'), preds_for_metric)
            np.save(os.path.join(results_folder_path, 'true_scaled.npy'), trues_for_metric)
        
        # 实验二: 分析并保存权重分布 (在循环外)
        if self.args.do_exp2_weight_dist:
            print("实验二: 正在分析并保存权重分布...")
            all_dynamic_weights = np.concatenate(all_dynamic_weights, axis=0) # 合并所有批次
            # all_dynamic_weights shape: [total_samples, pred_len, num_base_models]

            # 1. 保存原始权重数据，方便后续多数据集对比
            dist_folder_path = os.path.join(analysis_folder_path, 'exp2/', self.args.meta_learner_type, self.args.data_name)
            os.makedirs(dist_folder_path, exist_ok=True)
            np.save(os.path.join(dist_folder_path, 'all_dynamic_weights.npy'), all_dynamic_weights)

            # 2. 绘制并保存分布直方图
            num_base_models = all_dynamic_weights.shape[2]
            # 将所有样本和时间步的权重展平进行统计
            weights_flat = all_dynamic_weights.reshape(-1, num_base_models)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for k in range(num_base_models):
                ax.hist(weights_flat[:, k], bins=50, alpha=0.7, label=f'{self.args.base_model_names[k]} Weight')

            ax.set_title(f'Distribution of Learned Weights ({self.args.data_name})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

            hist_save_path = os.path.join(dist_folder_path, 'weights_histogram.pdf')
            plt.savefig(hist_save_path)
            plt.close(fig)
            print(f"权重分布图已保存至: {hist_save_path}")
        # =========================================================
        return