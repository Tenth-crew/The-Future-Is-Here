import json
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

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
        # 可以在这里为新的参数设置默认值，如果它们没有在 args 中提供的话
        if not hasattr(args, 'meta_hidden_dim'):
            args.meta_hidden_dim = 64 # 例如，默认的元模型隐藏层维度
        if not hasattr(args, 'meta_dropout_rate'):
            args.meta_dropout_rate = 0.0 # 默认不使用dropout

        super(Exp_stacking, self).__init__(args)


    def _override_args_for_base_model(self, shared_args, model_specific_params_dict):
        """
        辅助函数，用模型特定参数覆盖共享参数来创建新的参数命名空间。
        """
        # 创建共享参数字典的副本
        final_params_dict = vars(shared_args).copy()
        # 用模型特定参数更新（并覆盖）共享参数
        final_params_dict.update(model_specific_params_dict)
        return SimpleNamespace(**final_params_dict)

    def _build_model(self):
        args = self.args  # 主共享参数

        self.base_models = nn.ModuleList() # 使用 ModuleList 来正确注册模块

        # 检查必要的参数是否存在
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

            if model_name not in self.model_dict:
                raise ValueError(f"模型名称 '{model_name}' 在 model_dict 中未找到。可用模型: {list(self.model_dict.keys())}")

            print(f"构建基模型: {model_name}") # Debugging
            model_instance = self.model_dict[model_name].Model(current_model_args).to(self.device)
            self.base_models.append(model_instance)
        
        print(f"实际构建的基模型数量: {len(self.base_models)}")

        # 确定经过f_dim切片后每个基模型的输出特征维度和目标特征维度
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

        num_active_base_models = len(self.base_models)
        if num_active_base_models == 0:
            raise ValueError("没有指定或初始化任何基模型，无法进行stacking。")

        meta_model_input_dim = num_active_base_models * num_features_out_per_base_model
        
        if meta_model_input_dim == 0 :
             raise ValueError(f"计算得到的元模型输入维度 meta_model_input_dim 为 0 (基模型数量: {num_active_base_models}, "
                              f"每个基模型输出特征数: {num_features_out_per_base_model})。"
                              f"请检查 'c_out' 和 'features' 参数设置。")

        # 元模型定义 (MLP结构，维度动态计算)

        self.meta_model = nn.Sequential(
            nn.Linear(meta_model_input_dim, target_feature_dim),
            # # nn.BatchNorm1d(args.meta_hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(p=args.meta_dropout_rate) if args.meta_dropout_rate > 0 else nn.Identity(),
            # nn.LayerNorm(args.meta_hidden_dim),
            # nn.Linear(args.meta_hidden_dim, target_feature_dim)    
        ).to(self.device)
        
        print(f"元模型结构: {self.meta_model}")


        # 多GPU处理
        meta_model_to_return = self.meta_model
        if args.use_multi_gpu and args.use_gpu and torch.cuda.device_count() > 1:
            print(f"使用多GPU: {args.device_ids}")
            for i in range(len(self.base_models)):
                self.base_models[i] = nn.DataParallel(self.base_models[i], device_ids=args.device_ids)
            meta_model_to_return = nn.DataParallel(self.meta_model, device_ids=args.device_ids)
            
        return meta_model_to_return

    def _get_data(self, flag):
        # 此方法保持不变
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        all_params = []
        # 添加所有活动基模型的参数
        for model in self.base_models:
            all_params.extend(list(model.parameters()))
        
        # 添加元模型 (self.model) 的参数
        all_params.extend(list(self.model.parameters()))
        
        if not all_params:
            raise ValueError("优化器没有收到任何参数。请检查模型定义。")

        model_optim = optim.Adam(all_params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 此方法保持不变
        criterion = nn.MSELoss()
        return criterion
    

    def _process_batch(self, batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim):
        """辅助函数，处理一个批次数据，获取基模型输出并拼接。"""
        dec_inp = torch.zeros_like(batch_y_original[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y_original[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        base_model_outputs_raw = []
        for model_instance in self.base_models:
            if self.args.output_attention: # 假设 output_attention 适用于基模型
                output = model_instance(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                output = model_instance(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            base_model_outputs_raw.append(output)

        base_model_outputs_sliced = []
        for raw_output in base_model_outputs_raw:
            sliced_output = raw_output[:, -self.args.pred_len:, f_dim:]
            if self.args.features == 'MS' and sliced_output.ndim == 2:
                sliced_output = sliced_output.unsqueeze(-1)
            base_model_outputs_sliced.append(sliced_output)
        
        if not base_model_outputs_sliced: # 如果没有基模型（理论上_build_model会阻止这种情况）
            raise RuntimeError("没有基模型输出可以拼接。")

        stacked_features = torch.cat(base_model_outputs_sliced, dim=-1)
        return stacked_features

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test') # 你在训练时也加载了测试数据

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 设置所有模型为训练模式
            for model in self.base_models:
                model.train()
            self.model.train() # 元模型
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y_original = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_final_target = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.features == 'MS' and batch_y_final_target.ndim == 2:
                     batch_y_final_target = batch_y_final_target.unsqueeze(-1)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        stacked_features = self._process_batch(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                        final_outputs = self.model(stacked_features)
                        loss = criterion(final_outputs, batch_y_final_target)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    stacked_features = self._process_batch(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                    final_outputs = self.model(stacked_features)
                    loss = criterion(final_outputs, batch_y_final_target)
                    loss.backward()
                    model_optim.step()
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            
            epoch_train_loss_avg = np.average(train_loss)
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")
            
            print("Validating...")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # 你的代码中用 vali 方法在测试集上计算损失，并命名为 test_loss
            # 这意味着 vali 方法需要足够通用，或者你可能想为 test_loss 单独写一个评估逻辑
            test_loss_during_train = self.vali(test_data, test_loader, criterion) 

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {epoch_train_loss_avg:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss (during train): {test_loss_during_train:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, data_set, data_loader_val, criterion): # 修改参数名以反映其通用性
        total_loss = []
        # 设置所有模型为评估模式
        for model in self.base_models:
            model.eval()
        self.model.eval() # 元模型
        
        f_dim = -1 if self.args.features == 'MS' else 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader_val):
                batch_x = batch_x.float().to(self.device)
                batch_y_original = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_y_final_target = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.features == 'MS' and batch_y_final_target.ndim == 2:
                     batch_y_final_target = batch_y_final_target.unsqueeze(-1)

                stacked_features = self._process_batch(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                final_outputs = self.model(stacked_features)
                loss = criterion(final_outputs, batch_y_final_target)
                total_loss.append(loss.item())
        
        avg_loss = np.average(total_loss)
        # 不要在 vali 方法结束时将模型切换回 train() 模式，这应该由主训练循环在每个epoch开始时处理。
        return avg_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model for testing...')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            # Fallback (与你代码一致)
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds_for_metric, trues_for_metric = [], []
        preds_inv, trues_inv = [], [] # 用于存储可能逆变换后的结果

        results_folder_path = './results/' + setting + '/'
        if not os.path.exists(results_folder_path): os.makedirs(results_folder_path)
        visualization_folder_path = './test_results/' + setting + '/'
        if not os.path.exists(visualization_folder_path): os.makedirs(visualization_folder_path)

        # 设置所有模型为评估模式
        for model in self.base_models:
            model.eval()
        self.model.eval() # 元模型

        f_dim = -1 if self.args.features == 'MS' else 0

        with torch.no_grad():
            for i, (batch_x, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_original = batch_y_original.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                stacked_features = self._process_batch(batch_x, batch_y_original, batch_x_mark, batch_y_mark, f_dim)
                final_outputs = self.model(stacked_features)

                target_y = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.features == 'MS' and target_y.ndim == 2:
                    target_y = target_y.unsqueeze(-1)

                numpy_outputs_scaled = final_outputs.detach().cpu().numpy()
                numpy_target_scaled = target_y.detach().cpu().numpy()

                preds_for_metric.append(numpy_outputs_scaled)
                trues_for_metric.append(numpy_target_scaled)
                
                current_preds_to_save = numpy_outputs_scaled
                current_trues_to_save = numpy_target_scaled

                if test_data.scale and self.args.inverse:
                    shape_out = numpy_outputs_scaled.shape
                    current_preds_to_save = test_data.inverse_transform(numpy_outputs_scaled.reshape(shape_out[0] * shape_out[1], -1)).reshape(shape_out)
                    shape_true = numpy_target_scaled.shape
                    current_trues_to_save = test_data.inverse_transform(numpy_target_scaled.reshape(shape_true[0] * shape_true[1], -1)).reshape(shape_true)
                
                # 你原来的代码在逆变换后又进行了一次切片，这取决于 inverse_transform 的行为
                # 如果 inverse_transform 后维度恢复到 f_dim 处理前，则需要再次切片
                # 如果 inverse_transform 作用于已切片的数据且保持维度，则不需要
                # 这里保留你原来的逻辑
                current_preds_to_save = current_preds_to_save[:, -self.args.pred_len:, f_dim:]
                current_trues_to_save = current_trues_to_save[:, -self.args.pred_len:, f_dim:]
                
                preds_inv.append(current_preds_to_save)
                trues_inv.append(current_trues_to_save)

                if i % 20 == 0:
                    input_viz = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                         shape_in_viz = input_viz.shape
                         try:
                            input_viz = test_data.inverse_transform(input_viz.reshape(shape_in_viz[0] * shape_in_viz[1], -1)).reshape(shape_in_viz)
                         except Exception as e:
                            print(f"Warning: Could not inverse transform input_viz for visualization: {e}")
                    
                    num_features_to_plot = current_trues_to_save.shape[-1]
                    for feature_idx in range(num_features_to_plot):
                        # 确保 input_viz[:, :, feature_idx] 是有效的
                        # 如果 input_viz 的特征数与 current_trues_to_save 不同，这里可能需要调整
                        # 例如，如果 MS 模式下 current_trues_to_save 只有一个特征 (feature_idx=0),
                        # 你可能想从 input_viz 中选择一个对应的特征（比如第0个或最后一个）
                        input_hist_for_plot = input_viz[0, :, feature_idx if input_viz.shape[-1] > feature_idx else 0]

                        gt_viz = np.concatenate((input_hist_for_plot, current_trues_to_save[0, :, feature_idx]), axis=0)
                        pd_viz = np.concatenate((input_hist_for_plot, current_preds_to_save[0, :, feature_idx]), axis=0)
                        visual(gt_viz, pd_viz, os.path.join(visualization_folder_path, f"{i}_batch_sample0_feature{feature_idx}.pdf"))

        preds_for_metric = np.concatenate(preds_for_metric, axis=0)
        trues_for_metric = np.concatenate(trues_for_metric, axis=0)
        preds_inv = np.concatenate(preds_inv, axis=0)
        trues_inv = np.concatenate(trues_inv, axis=0)

        print('Test shapes (after concat):')
        print(f'preds_for_metric shape: {preds_for_metric.shape} | trues_for_metric shape: {trues_for_metric.shape}')
        preds_for_metric = preds_for_metric.reshape(-1, preds_for_metric.shape[-2], preds_for_metric.shape[-1])
        trues_for_metric = trues_for_metric.reshape(-1, trues_for_metric.shape[-2], trues_for_metric.shape[-1])
        print('Test shapes (after potential reshape):', preds_for_metric.shape, trues_for_metric.shape)

        mae, mse, rmse, mape, mspe = metric(preds_for_metric, trues_for_metric)
        
        dtw_val = -999
        if self.args.use_dtw:
            try:
                from utils.dtw_metric import accelerated_dtw # 确保导入
                dtw_list = []
                for k in range(preds_for_metric.shape[0]):
                    # 假设对第一个特征计算DTW
                    x_dtw = preds_for_metric[k, :, 0].reshape(-1, 1) 
                    y_dtw = trues_for_metric[k, :, 0].reshape(-1, 1)
                    manhattan_distance = lambda x, y: np.abs(x - y)
                    d, _, _, _ = accelerated_dtw(x_dtw, y_dtw, dist=manhattan_distance)
                    dtw_list.append(d)
                if dtw_list: dtw_val = np.array(dtw_list).mean()
            except ImportError: print("Warning: `accelerated_dtw` not found. Skipping DTW.")
            except Exception as e: print(f"Error during DTW: {e}. Skipping.")

        print(f'Metrics on scaled data: MSE:{mse:.4f}, MAE:{mae:.4f}, DTW:{dtw_val:.4f}')

        # 新增一个函数或直接放在 test() 函数末尾
        result_file = 'experiment_results_stacking.csv'
        args = self.args

        # 构建要保存的关键参数和指标
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': args.data_path,
            'base_model_names': str(args.base_model_names),
            'meta_hidden_dim': args.meta_hidden_dim,
            'meta_dropout_rate': args.meta_dropout_rate,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }

        # 将记录写入 Excel 表格，按 base_model_names 分 sheet
        df_record = pd.DataFrame([record])
        mode = 'w' if not os.path.exists(result_file) else 'a'
        df_record.to_csv(result_file, mode=mode, header=not os.path.exists(result_file), index=False)
        
        with open("result_long_term_forecast.txt", 'a') as f_res:
            f_res.write(f"{setting}\n")
            f_res.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, dtw:{dtw_val:.4f}\n\n')

        np.save(os.path.join(results_folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, dtw_val]))
        np.save(os.path.join(results_folder_path, 'pred_scaled.npy'), preds_for_metric)
        np.save(os.path.join(results_folder_path, 'true_scaled.npy'), trues_for_metric)
        if test_data.scale and self.args.inverse:
            np.save(os.path.join(results_folder_path, 'pred_inverse.npy'), preds_inv)
            np.save(os.path.join(results_folder_path, 'true_inverse.npy'), trues_inv)
        else:
            np.save(os.path.join(results_folder_path, 'pred_final.npy'), preds_inv) 
            np.save(os.path.join(results_folder_path, 'true_final.npy'), trues_inv)
        return