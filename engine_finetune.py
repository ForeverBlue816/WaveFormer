import os

import math
import sys
from typing import Iterable, Optional

import torch

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import r_regression

import wandb

import umap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from timm.data import Mixup

import util.misc as misc
import util.lr_sched as lr_sched
import util.plot as plot


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    training_history = {}
    
    # required for metrics calculation
    logits, labels = [], []

    for data_iter_step, (samples, targets, targets_mask, pos_embed_y) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets * targets_mask
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        if args.downstream_task == 'classification':
            targets_mask = targets_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)
        #elif args.downstream_task == 'regression':
            #print(f"[调试] 回归任务: 初始 targets 形状 {targets.size()}, targets_mask 形状 {targets_mask.size()}")
            #if targets.size() != targets_mask.size():
                #print(f"[修复] 回归任务: 修复前 targets_mask 形状 {targets_mask.size()} 应为 {targets.size()}")
                #targets_mask = torch.ones_like(targets, dtype=torch.float32, device=device)
                #print(f"[修复] 修复后 targets_mask 形状 {targets_mask.size()}")
            #else:
                #print(f"[验证通过] targets_mask 已正确初始化，形状为 {targets_mask.size()}")
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # 检查和初始化 targets_mask
        #if targets_mask is None or targets_mask.size(1) == 0:
            #print("警告: targets_mask 未初始化，自动初始化为全1张量。")
            #targets_mask = torch.ones_like(targets, dtype=torch.float32, device=device)

        with torch.amp.autocast(device_type='cuda'):
            #outputs = model(samples, pos_embed_y)

             # 检查和修复形状不匹配
            #if outputs.size() != targets_mask.size():
                 #print(f"警告: outputs 大小 {outputs.size()} 与 targets_mask 大小 {targets_mask.size()} 不匹配，自动调整。")
                 #targets_mask = torch.ones_like(outputs, dtype=torch.float32, device=device)
            outputs = model(samples, pos_embed_y) * targets_mask
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value) and misc.is_main_process():
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        logits.append(outputs)
        labels.append(targets)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    logits = torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()    # (B, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=-1)                                 # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()                         # (B, 1)
    
    training_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.downstream_task == 'classification':
        # Ensure num_classes is properly specified
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=args.nb_classes)  # (B, num_classes)


        f1 = 100*f1_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")
        precision = 100*precision_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")   # macro
        recall = 100*recall_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")         # macro
        acc = 100*accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
        acc_balanced = 100*balanced_accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
        auc = 100*roc_auc_score(y_true=labels_onehot, y_score=probs, average="macro",multi_class='ovr')                   # macro
        auprc = 100*average_precision_score(y_true=labels_onehot, y_score=probs, average="macro")

        training_stats["f1"] = f1
        training_stats["precision"] = precision
        training_stats["recall"] = recall
        training_stats["acc"] = acc
        training_stats["acc_balanced"] = acc_balanced
        training_stats["auroc"] = auc
        training_stats["auprc"] = auprc
    elif args.downstream_task == 'regression':
        # 打印 logits 和 labels 的形状
        #print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
        #if logits.ndim > 2:
            #logits = logits.view(-1, logits.shape[-1])  # 展开时间步或多维度
        #if labels.ndim > 2:
            #labels = labels.view(-1, labels.shape[-1])  # 展开时间步或多维度
        rmse = np.float64(root_mean_squared_error(logits, labels, multioutput="raw_values"))
        training_stats["rmse"] = rmse if isinstance(rmse, float) else rmse.mean(axis=-1)

        mae = np.float64(mean_absolute_error(logits, labels, multioutput="raw_values"))
        training_stats["mae"] = mae if isinstance(mae, float) else mae.mean(axis=-1)

        pcc = np.concatenate([r_regression(logits[:, i].view(-1, 1), labels[:, i]) for i in range(labels.shape[-1])], axis=0)
        training_stats["pcc"] = pcc if isinstance(pcc, float) else pcc.mean(axis=-1)

        r2 = np.stack([r2_score(labels[:, i], logits[:, i]) for i in range(labels.shape[-1])], axis=0)
        training_stats["r2"] = r2 if isinstance(pcc, float) else r2.mean(axis=-1)

    # tensorboard
    if log_writer is not None: #and (data_iter_step + 1) % accum_iter == 0:
        #""" We use epoch_1000x as the x-axis in tensorboard.
        #This calibrates different curves when batch size changes.
        #"""
        #epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        log_writer.add_scalar('loss', training_stats["loss"], epoch)
        log_writer.add_scalar('lr', training_stats["lr"], epoch)

        if args.downstream_task == 'classification':
            log_writer.add_scalar('perf/train_f1', f1, epoch)
            log_writer.add_scalar('perf/train_precision', precision, epoch)
            log_writer.add_scalar('perf/train_recall', recall, epoch)
            log_writer.add_scalar('perf/train_acc', acc, epoch)
            log_writer.add_scalar('perf/train_acc_balanced', acc_balanced, epoch)
            log_writer.add_scalar('perf/train_auroc', auc, epoch)
            log_writer.add_scalar('perf/train_auprc', auprc, epoch)
        elif args.downstream_task == 'regression':
            log_writer.add_scalar('perf/train_rmse', training_stats["rmse"], epoch)
            log_writer.add_scalar('perf/train_mae', training_stats["mae"], epoch)
            log_writer.add_scalar('perf/train_pcc', training_stats["pcc"], epoch)
            log_writer.add_scalar('perf/train_r2', training_stats["r2"], epoch)

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history['loss'] = training_stats["loss"]
        training_history['lr'] = training_stats["lr"]
        if args.downstream_task == 'classification':
            training_history['f1'] = f1
            training_history['precision'] = precision
            training_history['recall'] = recall
            training_history['acc'] = acc
            training_history['acc_balanced'] = acc_balanced
            training_history['auroc'] = auc
            training_history['auprc'] = auprc
        elif args.downstream_task == 'regression':
            training_history['rmse'] = training_stats["rmse"]
            training_history['mae'] = training_stats["mae"]
            training_history['pcc'] = training_stats["pcc"]
            training_history['r2'] = training_stats["r2"]

            if targets.shape[-1] > 1:
                for i in range(targets.shape[-1]):
                    training_history[f'Train/RMSE/{i}'] = rmse[i]
                    training_history[f'Train/MAE/{i}'] = mae[i]
                    training_history[f'Train/PCC/{i}'] = pcc[i]
                    training_history[f'Train/R2/{i}'] = r2[i]

    return training_stats, training_history


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()

    if args.downstream_task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    test_history = {}
    
    # required for metrics calculation
    embeddings, logits, labels = [], [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        images = images.to(device, non_blocking=True)

        target = batch[1]
        target = target.to(device, non_blocking=True)

        target_mask = batch[2]
        target_mask = target_mask.to(device, non_blocking=True)
        target = target * target_mask

        pos_embed_y = batch[3]
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        if args.downstream_task == 'classification':
            target_mask = target_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        # compute output
        with torch.amp.autocast(device_type='cuda'):
            embedding = model.forward_features(images, pos_embed_y)
            output = model.forward_head(embedding)
            output = output * target_mask
            loss = criterion(output, target)

        if args.save_embeddings:
            embeddings.append(embedding)
        logits.append(output)
        labels.append(target)

        metric_logger.update(loss=loss.item())

    if args.wandb and args.plot_attention_map:
        attention_map = model.blocks[-1].attn.attn_map
        idx = 1 if args.batch_size > 1 else 0
        test_history["Attention"] = plot.plot_attention(images, attention_map, idx)

    if args.save_embeddings and misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach() # (B, D)
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_test.pt" if args.eval else f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes
    if epoch != -1:
        metric_logger.synchronize_between_processes()

    logits = torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()    # (B, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=-1)                                 # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()                         # (B, 1)
    
    if args.save_logits and misc.is_main_process():
        logits_path = os.path.join(args.output_dir, "logits")
        if not os.path.exists(logits_path):
            os.makedirs(logits_path)
        
        file_name = f"logits_test.pt" if args.eval else f"logits_{epoch}.pt"
        torch.save(logits, os.path.join(logits_path, file_name))

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.downstream_task == 'classification':
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=args.nb_classes)                 # (B, num_classes)
        f1 = 100*f1_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")
        precision = 100*precision_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")    # macro
        recall = 100*recall_score(y_true=labels, y_pred=logits.argmax(dim=-1), average="macro")          # macro
        acc = 100*accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
        acc_balanced = 100*balanced_accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
        auc = 100*roc_auc_score(y_true=labels_onehot, y_score=probs, average="macro",multi_class='ovr')                    # macro
        auprc = 100*average_precision_score(y_true=labels_onehot, y_score=probs, average="macro")
        
        test_stats["f1"] = f1
        test_stats["precision"] = precision
        test_stats["recall"] = recall
        test_stats["acc"] = acc
        test_stats["acc_balanced"] = acc_balanced
        test_stats["auroc"] = auc
        test_stats["auprc"] = auprc
    elif args.downstream_task == 'regression':
        # 打印 logits 和 labels 的形状
        #print(f"Before reshaping - logits shape: {logits.shape}, labels shape: {labels.shape}")

        # 展开维度
        #if logits.ndim == 3:
            #logits = logits.reshape(-1, logits.shape[-1])  # (batch_size * sequence_length, num_classes)
        #if labels.ndim == 3:
            #labels = labels.reshape(-1, labels.shape[-1])  # (batch_size * sequence_length, num_classes)
        # 打印调整后的形状
        #print(f"After reshaping - logits shape: {logits.shape}, labels shape: {labels.shape}")

        rmse = np.float64(root_mean_squared_error(logits, labels, multioutput="raw_values"))
        test_stats["rmse"] = rmse if isinstance(rmse, float) else rmse.mean(axis=-1)

        mae = np.float64(mean_absolute_error(logits, labels, multioutput="raw_values"))
        test_stats["mae"] = mae if isinstance(mae, float) else mae.mean(axis=-1)

        pcc = np.concatenate([r_regression(logits[:, i].view(-1, 1), labels[:, i]) for i in range(labels.shape[-1])], axis=0)
        test_stats["pcc"] = pcc if isinstance(pcc, float) else pcc.mean(axis=-1)

        r2 = np.stack([r2_score(labels[:, i], logits[:, i]) for i in range(labels.shape[-1])], axis=0)
        test_stats["r2"] = r2 if isinstance(r2, float) else r2.mean(axis=-1)

    if args.downstream_task == 'classification':
        print('* Acc@1 {top1_acc:.2f} Acc@1 (balanced) {acc_balanced:.2f} Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f} AUROC {auroc:.2f} AUPRC {auprc:.2f} loss {losses:.3f}'
              .format(top1_acc=acc, acc_balanced=acc_balanced, precision=precision, recall=recall, f1=f1, auroc=auc, auprc=auprc, losses=test_stats["loss"]))
    elif args.downstream_task == 'regression':
        print('* RMSE {rmse:.3f} MAE {mae:.3f} PCC {pcc:.3f} R2 {r2:.3f} loss {losses:.3f}'
              .format(rmse=test_stats["rmse"], mae=test_stats["mae"], pcc=test_stats["pcc"], r2=test_stats["r2"], losses=test_stats["loss"]))

    # tensorboard
    if log_writer is not None:
        if args.downstream_task == 'classification':
            log_writer.add_scalar('perf/test_f1', f1, epoch)
            log_writer.add_scalar('perf/test_precision', precision, epoch)
            log_writer.add_scalar('perf/test_recall', recall, epoch)
            log_writer.add_scalar('perf/test_acc', acc, epoch)
            log_writer.add_scalar('perf/test_acc_balanced', acc_balanced, epoch)
            log_writer.add_scalar('perf/test_auroc', auc, epoch)
            log_writer.add_scalar('perf/test_auprc', auprc, epoch)
        elif args.downstream_task == 'regression':
            log_writer.add_scalar('perf/test_rmse', test_stats['rmse'], epoch)
            log_writer.add_scalar('perf/test_mae', test_stats['mae'], epoch)
            log_writer.add_scalar('perf/test_pcc', test_stats['pcc'], epoch)
            log_writer.add_scalar('perf/test_r2', test_stats['r2'], epoch)
        log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

    # wandb
    if args.wandb == True:
        test_history = {'epoch' : epoch, 'test_loss' : test_stats['loss']}
        if args.downstream_task == 'classification':
            test_history['test_f1'] = f1
            test_history['test_precision'] = precision
            test_history['test_recall'] = recall
            test_history['test_acc'] = acc
            test_history['test_acc_balanced'] = acc_balanced
            test_history['test_auroc'] = auc
            test_history['test_auprc'] = auprc
        elif args.downstream_task == 'regression':
            test_history['test_rmse'] = test_stats['rmse']
            test_history['test_mae'] = test_stats['mae']
            test_history['test_pcc'] = test_stats['pcc']
            test_history['test_r2'] = test_stats['r2']

            if target.shape[-1] > 1:
                for i in range(target.shape[-1]):
                    test_history[f'Test/RMSE/{i}'] = rmse[i]
                    test_history[f'Test/MAE/{i}'] = mae[i]
                    test_history[f'Test/PCC/{i}'] = pcc[i]
                    test_history[f'Test/R2/{i}'] = r2[i]

        if args.plot_embeddings and epoch % 10 == 0:
            reducer = umap.UMAP(n_components=2, metric='euclidean')
            umap_proj = reducer.fit_transform(embeddings)
            
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = matplotlib.cm.get_cmap('tab20') # for the colours
            for label in range(args.nb_classes):
                indices = labels.numpy()==label
                ax.scatter(umap_proj[indices, 0], umap_proj[indices, 1], c=np.array(cmap(label*3)).reshape(1, 4), label=label, alpha=0.5)

            ax.legend(fontsize='large', markerscale=2)

            test_history["UMAP Embeddings"] = wandb.Image(fig)
            plt.close('all')
    
    return test_stats, test_history
