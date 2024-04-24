import random
import time
import subprocess
import traceback
import socket
import setproctitle

from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.optim
import torch.utils.data
import copy
import logging
import os
import re
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import datetime

from utils.commons.ckpt_utils import get_last_checkpoint, get_all_ckpts
from utils.commons.ddp_utils import DDP
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import move_to_cuda
from utils.commons.os_utils import remove_file
from utils.commons.meters import Timer


def check_port_is_occupied(host='localhost', port=10080):
    s = socket.socket()
    try:
        s.connect((host, port))
        print(f"{host}:{port} is occupied!")
        return True
    except:
        print(f"{host}:{port} is not occupied!")
        return False
    finally:
        s.close()


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class Trainer:
    def __init__(
            self,
            work_dir,
            default_save_path=None,
            accumulate_grad_batches=1,
            max_updates=160000,
            print_nan_grads=False,
            val_check_interval=2000,
            num_sanity_val_steps=5,
            amp=False,
            # tb logger
            log_save_interval=100,
            tb_log_interval=10,
            # checkpoint
            monitor_key='val_loss',
            monitor_mode='min',
            num_ckpt_keep=5,
            save_best=True,
            resume_from_checkpoint=0,
            seed=1234,
            debug=False,
    ):
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_updates = max_updates
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.default_save_path = default_save_path
        self.resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint > 0 else None
        self.seed = seed
        self.debug = debug
        # model and optm
        self.task = None
        self.optimizers = []

        # trainer state
        self.testing = False
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure checkpoint
        self.monitor_key = monitor_key
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.monitor_op = np.less if monitor_mode == 'min' else np.greater
        self.best_val_results = np.Inf if monitor_mode == 'min' else -np.Inf
        self.mode = 'min'

        # allow int, string and gpu list
        self.all_gpu_ids = [
            int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
        
        self.use_multi_machine_ddp = hparams['world_size'] != -1
        self.num_local_gpus = len(self.all_gpu_ids) # if world_size is not -1, multi-machine setting
        self.num_total_gpus = len(self.all_gpu_ids) if not self.use_multi_machine_ddp else hparams['world_size'] # if world_size is not -1, multi-machine setting
        # self.num_gpus = len(self.all_gpu_ids) 
        self.on_gpu = self.num_local_gpus > 0
        self.root_gpu = 0
        logging.info(f'GPU available: {torch.cuda.is_available()}, GPU used: {self.all_gpu_ids}, world_size: {self.num_total_gpus}, multi-machine training: {self.use_multi_machine_ddp}')
        self.use_ddp = self.num_local_gpus > 1 or self.use_multi_machine_ddp
        self.proc_rank = 0
        # Tensorboard logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.tb_log_interval = tb_log_interval
        self.amp = amp
        self.amp_scalar = GradScaler()

    def test(self, task_cls):
        self.testing = True
        self.fit(task_cls)

    def fit(self, task_cls):
        try:
            if self.use_ddp:
                # mp.spawn(self.ddp_run, nprocs=self.num_local_gpus, args=(task_cls, copy.deepcopy(hparams)))
                mp.start_processes(self.ddp_run,nprocs=self.num_local_gpus, args=(task_cls, copy.deepcopy(hparams)), start_method='spawn')
            else:
                # File "/mnt/bn/ailabrenyi/entries/yezhenhui/projects/GeneFace_private/venv_113/lib/python3.9/site-packages/torch/nn/modules/batchnorm.py", line 735, in forward
                # world_size = torch.distributed.get_world_size(process_group)
                # to address the error in batchNorm using venv_113
                # self.init_ddp_connection_tcp(0, 1) 
                self.task = task_cls()
                self.task.trainer = self
                setproctitle.setproctitle(f'GeneFace_worker ({hparams["work_dir"]})')
                self.run_single_process(self.task)
        except:
            traceback.print_exc()
            time.sleep(5)
            # subprocess.check_call(f'pkill -f "GeneFace_worker \({hparams["work_dir"]}"', shell=True)禁止清理！
        return 1



    def ddp_run(self, gpu_idx, task_cls, hparams_):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'GeneFace_worker ({hparams_["work_dir"]}_{gpu_idx})')

        if hparams.get('use_file_system_mp'):
            torch.multiprocessing.set_sharing_strategy('file_system')
        if hparams.get('use_fork', True):
            torch.multiprocessing.set_start_method('fork', force=True)
        self.root_gpu = gpu_idx
        self.proc_rank = gpu_idx + hparams['start_rank'] if self.use_multi_machine_ddp else gpu_idx
        print("before init_tcp!")
        if hparams['init_method'] == 'file':
            self.init_ddp_connection_file(self.proc_rank, self.num_total_gpus)
        elif hparams['init_method'] == 'tcp':
            self.init_ddp_connection_tcp(self.proc_rank, self.num_total_gpus)
        else:
            raise NotImplementedError()

        if gpu_idx != 0 and not self.debug:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        dist.barrier()
        print("after init_tcp!")

        task = task_cls()
        task.trainer = self
        torch.cuda.set_device(gpu_idx)
        self.task = task
        self.run_single_process(task)

    def run_single_process(self, task):
        """Sanity check a few things before starting actual training.

        :param task:
        """
        # build model, optm and load checkpoint
        if self.proc_rank == 0:
            self.save_terminal_logs()       # 这个函数用来设定终端的log参数，例如训练的迭代次数
            if not self.testing:
                self.save_codes()

        model = task.build_model()      # 这里开始初始化模型
        if model is not None:
            task.model = model
        checkpoint, _ = get_last_checkpoint(self.work_dir, self.resume_from_checkpoint)
        if checkpoint is not None:
            self.restore_weights(checkpoint)
        elif self.on_gpu:
            task.cuda(self.root_gpu)
        if not self.testing:
            self.optimizers = task.configure_optimizers()
            self.fisrt_epoch = True
        if checkpoint is not None:
            self.restore_opt_state(checkpoint)
        del checkpoint
        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        if self.use_ddp:
            self.task = self.configure_ddp(self.task)
            dist.barrier()

        task_ref = self.get_task_ref()
        task_ref.trainer = self
        task_ref.testing = self.testing
        # link up experiment object
        if self.proc_rank == 0:
            task_ref.build_tensorboard(save_dir=self.work_dir, name='tb_logs')
        else:
            os.makedirs('tmp', exist_ok=True)
            task_ref.build_tensorboard(save_dir='tmp', name='tb_tmp')
        self.logger = task_ref.logger
        try:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()
        except:
            traceback.print_exc()
            task_ref.on_keyboard_interrupt()
            time.sleep(5)
            # if self.proc_rank == 0:
            #     subprocess.check_call(f'pkill -f "GeneFace_worker \({hparams["work_dir"]}"', shell=True)

    ####################
    # valid and test
    ####################
    def run_evaluation(self, test=False):
        eval_results = self.evaluate(self.task, test, tqdm_desc='Valid' if not test else 'test',
                                     max_batches=hparams['eval_max_batches'])
        if eval_results is not None and 'tb_log' in eval_results:
            tb_log_output = eval_results['tb_log']
            self.log_metrics_to_tb(tb_log_output)
        if self.proc_rank == 0 and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        if max_batches == -1:
            max_batches = None
        # enable eval mode
        task.zero_grad()
        task.eval()
        torch.set_grad_enabled(False)

        task_ref = self.get_task_ref()
        if test:
            ret = task_ref.test_start()
            if ret == 'EXIT':
                return
        else:
            task_ref.validation_start()
        outputs = []
        dataloader = task_ref.test_dataloader() if test else task_ref.val_dataloader()
        pbar = tqdm.tqdm(dataloader, desc=tqdm_desc, total=max_batches, dynamic_ncols=True, unit='step',
                         disable=self.root_gpu > 0)
        # give model a chance to do something with the outputs (and method defined)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:  # pragma: no cover
                continue
            # stop short when on fast_dev_run (sets max_batch=1)
            if max_batches is not None and batch_idx >= max_batches:
                break

            # make dataloader_idx arg in validation_step optional
            if self.on_gpu:
                batch = move_to_cuda(batch, self.root_gpu)
            args = [batch, batch_idx]
            if self.use_ddp:
                output = task(*args)
            else:
                if test:
                    output = task_ref.test_step(*args)
                else:
                    output = task_ref.validation_step(*args)
            # track outputs for collation
            outputs.append(output)
        # give model a chance to do something with the outputs (and method defined)
        if test:
            eval_results = task_ref.test_end(outputs)
        else:
            eval_results = task_ref.validation_end(outputs)
        # enable train mode again
        task.train()
        torch.set_grad_enabled(True)
        return eval_results

    ####################
    # train
    ####################
    def train(self):        # 真正的训练入口
        task_ref = self.get_task_ref()
        task_ref.on_train_start()
        if self.num_sanity_val_steps > 0:
            # run tiny validation (if validation defined) to make sure program won't crash during val
            self.evaluate(self.task, False, 'Sanity Val', max_batches=self.num_sanity_val_steps)
        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()
        dataloader = task_ref.train_dataloader()
        epoch = self.current_epoch
        # run all epochs
        while True:
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            # update training progress in trainer and model
            task_ref.current_epoch = epoch
            self.current_epoch = epoch
            # total batches includes multiple val checks
            self.batch_loss_value = 0  # accumulated grads
            # before epoch hook
            task_ref.on_epoch_start()

            # run epoch
            train_pbar = tqdm.tqdm(dataloader, initial=self.global_step, total=float('inf'),
                                   dynamic_ncols=True, unit='step', disable=self.root_gpu > 0)
            # for batch_idx, batch in enumerate(train_pbar):
            train_iterator = iter(enumerate(train_pbar))
            while True:
                with Timer("get_batch", enable=self.debug):
                    try:
                        batch_idx, batch = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(enumerate(train_pbar))
                        batch_idx, batch = next(train_iterator)

                if self.global_step % self.val_check_interval == 0 and not self.fisrt_epoch:
                    self.run_evaluation()
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                train_pbar.set_postfix(**pbar_metrics)
                self.fisrt_epoch = False
                # when metrics should be logged
                if (self.global_step + 1) % self.tb_log_interval == 0:
                    # logs user requested information to logger
                    self.log_metrics_to_tb(tb_metrics)

                self.global_step += 1
                task_ref.global_step = self.global_step
                if self.global_step > self.max_updates:
                    print("| Training end..")
                    break
            # epoch end hook
            epoch_loss_dict = task_ref.on_epoch_end()
            self.log_metrics_to_tb(epoch_loss_dict)
            epoch += 1
            if self.global_step > self.max_updates:
                break
        task_ref.on_train_end()

    def run_training_batch(self, batch_idx, batch):     # 正式开始训练
        if batch is None:
            return {}
        all_progress_bar_metrics = []
        all_log_metrics = []
        task_ref = self.get_task_ref()
        for opt_idx, optimizer in enumerate(self.optimizers):
            if optimizer is None:
                continue
            # make sure only the gradients of the current optimizer's paramaters are calculated
            # in the training step to prevent dangling gradients in multiple-optimizer setup.
            if len(self.optimizers) > 1:
                for k, param in task_ref.named_parameters():
                    param.requires_grad = False
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.requires_grad = True

            # forward pass
            with Timer("forward_training_step", enable=self.debug):
                with autocast(enabled=self.amp):
                    if self.on_gpu:
                        batch = move_to_cuda(copy.copy(batch), self.root_gpu)
                    args = [batch, batch_idx, opt_idx]
                    if self.use_ddp:
                        output = self.task(*args)
                    else:
                        output = task_ref.training_step(*args)      # 那其实这个训练入口是为了打开这个入口，麻烦得一笔啊
                    loss = output['loss']
                    if loss is None:
                        continue
                    progress_bar_metrics = output['progress_bar']
                    log_metrics = output['tb_log']
                    # accumulate loss
                    loss = loss / self.accumulate_grad_batches
                
            # backward pass
            with Timer("backward_training_step", enable=self.debug):
                if loss.requires_grad:
                    if self.amp:
                        self.amp_scalar.scale(loss).backward()
                    else:
                        loss.backward()

                # track progress bar metrics
                all_log_metrics.append(log_metrics)
                all_progress_bar_metrics.append(progress_bar_metrics)

                if loss is None:
                    continue

            # nan grads
            with Timer("checkNan_training_step", enable=self.debug):
                has_nan_grad = False
                nan_params_names = []
                if self.print_nan_grads:
                    for name, param in task_ref.named_parameters():
                        if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                            print("| NaN params: ", name, param, param.grad)
                            has_nan_grad = True
                            nan_params_names.append(name)
                    if has_nan_grad:
                        # exit(0)
                        print(f"| WARN: found nan in grad! first nan params: {nan_params_names[0]}; last nan params: {nan_params_names[-1]}.")
                        pass

            # gradient update with accumulated gradients
            with Timer("optimUpdate_training_step", enable=self.debug):
                if (self.global_step + 1) % self.accumulate_grad_batches == 0 and not has_nan_grad:
                # if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    if self.amp:
                        self.amp_scalar.unscale_(optimizer)
                    grad_norm_dict = task_ref.on_before_optimization(opt_idx)
                    if grad_norm_dict is not None:
                        all_log_metrics[-1].update(grad_norm_dict)
                    if self.amp:
                        self.amp_scalar.step(optimizer)
                        self.amp_scalar.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)

        # collapse all metrics into one dict
        all_progress_bar_metrics = {k: v for d in all_progress_bar_metrics for k, v in d.items()}
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        return all_progress_bar_metrics, all_log_metrics

    ####################
    # load and save checkpoint
    ####################
    def restore_weights(self, checkpoint):
        # load model state
        task_ref = self.get_task_ref()

        for k, v in checkpoint['state_dict'].items():
            if hasattr(task_ref, k):
                getattr(task_ref, k).load_state_dict(v, strict=True)
            else:
                print(f"| the checkpoint has unmatched keys {k}")

        if self.on_gpu:
            task_ref.cuda(self.root_gpu)
        # load training state (affects trainer only)
        self.best_val_results = checkpoint['checkpoint_callback_best']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        task_ref.global_step = self.global_step

        # wait for all models to restore weights
        if self.use_ddp:
            # wait for all processes to catch up
            dist.barrier()

    def restore_opt_state(self, checkpoint):
        if self.testing:
            return
        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            try:
                optimizer.load_state_dict(opt_state)
                # move optimizer to GPU 1 weight at a time
                if self.on_gpu:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.root_gpu)
            except ValueError:
                print("| WARMING: optimizer parameters not match !!!")
        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            print(e)
            return
        did_restore = True
        return did_restore

    def save_checkpoint(self, epoch, logs=None):
        monitor_op = np.less
        ckpt_path = f'{self.work_dir}/model_ckpt_steps_{self.global_step}.ckpt'
        logging.info(f'Epoch {epoch:05d}@{self.global_step}: saving model to {ckpt_path}')
        self._atomic_save(ckpt_path)

        get_ckpt_step_fn = lambda x: int(re.findall('.*steps\_(\d+)\.ckpt', x)[0])
        for old_ckpt in get_all_ckpts(self.work_dir)[self.num_ckpt_keep:]:
            # leave the milestone ckpts
            if hparams.get("ckpt_milestone_interval", 10_0000) != 0 and get_ckpt_step_fn(old_ckpt) % hparams.get("ckpt_milestone_interval", 10_0000) == 0:
                pass
            else:
                remove_file(old_ckpt)
                logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
        current = None
        if logs is not None and self.monitor_key in logs:
            current = logs[self.monitor_key]
        # if current is not None and self.save_best:
        #     if monitor_op(current, self.best_val_results):
        #         best_filepath = f'{self.work_dir}/model_ckpt_best.pt'
        #         self.best_val_results = current
        #         logging.info(
        #             f'Epoch {epoch:05d}@{self.global_step}: {self.monitor_key} reached {current:0.5f}. '
        #             f'Saving model to {best_filepath}')
        #         self._atomic_save(best_filepath)

    def _atomic_save(self, filepath):
        checkpoint = self.dump_checkpoint()
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, filepath)
    
    def dump_checkpoint(self):
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step,
                      'checkpoint_callback_best': self.best_val_results}
        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                state_dict = optimizer.state_dict()
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                optimizer_states.append(state_dict)

        checkpoint['optimizer_states'] = optimizer_states
        task_ref = self.get_task_ref()
        state_dict = {
            k: v.state_dict() for k, v in task_ref.named_children()
            if len(list(v.parameters())) > 0 and (k not in hparams.get('not_save_modules', []))
        }
        for module_k, module_k_dict in list(state_dict.items()):
            for k, v in list(module_k_dict.items()):
                if '_orig_mod.' in k:
                    module_k_dict[k.replace('_orig_mod.', '')] = v
                    del module_k_dict[k]
            
        checkpoint['state_dict'] = state_dict
        return checkpoint
    ####################
    # DDP
    ####################
    def configure_ddp(self, task):
        task = torch.nn.SyncBatchNorm.convert_sync_batchnorm(task)
        task = DDP(task, device_ids=[self.root_gpu], find_unused_parameters=True)
        # task = DDP(task, device_ids=[self.root_gpu], find_unused_parameters=False)
        random.seed(self.seed)
        np.random.seed(self.seed)
        return task

    def init_ddp_connection_file(self, proc_rank, world_size):
        """
        use a shared file in the network file system to bind all process
        if you found num_worker is larger than world_size, remove the old shard_file_name
        """
        exp_name = hparams['exp_name']
        shared_file_name = f'file:///home/tiger/nfs/pytorch_ddp_sharedfile/{exp_name}'
        os.makedirs(os.path.dirname(shared_file_name).replace("file://",""), exist_ok=True)
        dist.init_process_group('nccl', init_method=shared_file_name,
                            world_size=world_size, rank=proc_rank)
        
    def init_ddp_connection_tcp(self, proc_rank, world_size):
        if self.use_multi_machine_ddp:
            root_node, port = os.environ['ARNOLD_WORKER_HOSTS'].split(",")[0].split(":")
            os.environ['MASTER_PORT'] = '6668'
        else:
            root_node = '127.0.0.1'
        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node

        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size, timeout=datetime.timedelta(seconds=600))
        # dist.init_process_group('gloo', rank=proc_rank, world_size=world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]
            number = re.sub('[^0-9]', '', number)
            root_node = name + number
        return root_node

    ####################
    # utils
    ####################
    def get_task_ref(self):
        from utils.commons.base_task import BaseTask
        task: BaseTask = self.task.module if isinstance(self.task, DDP) else self.task
        return task

    def log_metrics_to_tb(self, metrics, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        """
        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0:
            self.log_metrics(self.logger, scalar_metrics, step=step)

    @staticmethod
    def log_metrics(logger, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def save_terminal_logs(self):   # 设定保存日志的路径，以及系统日期
        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs(f'{self.work_dir}/terminal_logs', exist_ok=True)
        Tee(f'{self.work_dir}/terminal_logs/log_{t}.txt', 'w')

    def save_codes(self):
        if len(hparams['save_codes']) > 0:
            t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = f'{self.work_dir}/codes/{t}'
            os.makedirs(code_dir, exist_ok=True)
            # subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            # for c in hparams['save_codes']:
            #     if os.path.exists(c):
            #         subprocess.check_call(
            #             f'rsync -aR '
            #             f'--include="*.py" '
            #             f'--include="*.yaml" '
            #             f'--exclude="__pycache__" '
            #             f'--include="*/" '
            #             f'--exclude="*" '
            #             f'"./{c}" "{code_dir}/"',
            #             shell=True)
            # print(f"| Copied codes to {code_dir}.")