# -*- coding: utf-8 -*-
import math
import random
import time
import csv
import os
import sys
from datetime import datetime
from collections import namedtuple, deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pymss  # type: ignore
from Model import SimulationNN, MuscleNN, ExoPolicyNN, use_cuda, FloatTensor, Tensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor  if use_cuda else torch.ByteTensor

Episode     = namedtuple('Episode',('s','a','r','value','logprob'))
Transition  = namedtuple('Transition',('s','a','logprob','TD','GAE'))

# ---------- 小工具：安全创建目录 ----------
def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class EpisodeBuffer(object):
    def __init__(self):
        self.data = []
    def Push(self, *args):
        self.data.append(Episode(*args))
    def Pop(self):
        if self.data:
            self.data.pop()
    def GetData(self):
        return self.data
    def Clear(self):
        self.data.clear()

class ReplayBuffer(object):
    def __init__(self, buff_size=10000):
        self.buffer = deque(maxlen=buff_size)
    def Push(self,*args):
        self.buffer.append(Transition(*args))
    def Clear(self):
        self.buffer.clear()

class PPO(object):
    """
    三网络协同：
      - SimulationNN：PD 策略 + 值函数
      - ExoPolicyNN：外骨骼策略（独立 policy head）
      - MuscleNN：肌肉自监督

    指标口径（与你刚才拍板一致）：
      - 策略相关（avg_tau_abs / sum_tau_sq / avg_activation / steps_in_episode）→【控制步】
      - max_tau_abs → 每个控制步内聚合细步峰值（若要严格“控制步瞬时”，可在注释处一行切换）
      - ctrl_avg_power / ctrl_energy → 统一从 C++ 回合向量读取
      - eval_round → 写 num_evaluation + 1
    """
    def __init__(self, meta_file):
        np.random.seed(seed=int(time.time()))
        self.num_slaves = 16
        self.env = pymss.pymss(meta_file, self.num_slaves)

        self.use_muscle   = self.env.UseMuscle()
        self.num_state    = self.env.GetNumState()
        self.num_action   = self.env.GetNumAction()
        self.num_muscles  = self.env.GetNumMuscles()
        # === 读取外骨骼 6 维动作空间（从 C++ 的缓存矩阵列数识别）===
        try:
            exo_mat = self.env.GetExoTorques()  # shape [num_envs, exo_cols]
            self.num_exo_action = exo_mat.shape[1]  # 应为 6
        except:
            self.num_exo_action = 6  # 兜底


        self.num_epochs          = 10
        self.num_epochs_muscle   = 3
        self.num_evaluation      = 0
        self.num_tuple_so_far    = 0
        self.num_episode         = 0
        self.num_tuple           = 0
        self.num_simulation_Hz   = self.env.GetSimulationHz()
        self.num_control_Hz      = self.env.GetControlHz()
        self.num_sim_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.gamma = 0.99
        self.lb    = 0.99

        self.buffer_size       = 2048
        self.batch_size        = 128
        self.muscle_batch_size = 128
        self.replay_buffer     = ReplayBuffer(30000)
        self.muscle_buffer     = {}

        # —— 三网络 —— #
        self.model       = SimulationNN(self.num_state, self.num_action)
        self.exo_model   = ExoPolicyNN(self.num_state, self.num_exo_action)

        self.muscle_model= MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),
                                     self.num_action, self.num_muscles)
        if use_cuda:
            self.model.cuda(); self.exo_model.cuda(); self.muscle_model.cuda()

        # —— 优化器（PD+Exo 合并；Muscle 单独） —— #
        self.learning_rate = 1e-4
        self.clip_ratio    = 0.2
        self.optimizer         = optim.Adam(list(self.model.parameters())+list(self.exo_model.parameters()),
                                            lr=self.learning_rate)
        self.optimizer_muscle  = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        self.max_iteration     = 50000
        self.w_entropy         = -0.001

        # —— 训练统计 —— #
        self.loss_actor  = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards     = []
        self.sum_return  = 0.0
        self.max_return  = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()

        # —— 每个并行环境的 episode 缓冲 —— #
        self.episodes = [EpisodeBuffer() for _ in range(self.num_slaves)]

        # ---------- 回合级指标累加器（全部在【控制步】层面更新一次） ----------
        def _new_ep_stat():
            return {
                # 力矩（有效 DOF 的 τ_des）
                'tau_abs_sum': 0.0,       # 累加：每控制步的 mean(|tau|)；最后 / step_count → avg_tau_abs
                'tau_sq_sum':  0.0,       # 累加：每控制步的 sum(tau^2) → sum_tau_sq
                'tau_abs_max': 0.0,       # 最大：回合 max(|tau|)（控制步内聚合细步峰值）
                # 激活（use_muscle=True 有效）
                'act_abs_sum': 0.0,       # 累加：每控制步的 mean(|a|)；最后 / step_count → avg_activation
                'act_sq_sum':  0.0,       # 累加：每控制步的 sum(a^2) → sum_activation_sq
                'act_abs_max': 0.0,       # 最大：回合 max(|a|)
                # 外骨骼力矩（Python 侧统计，写在扩展 CSV）
                'exo_abs_sum': 0.0,       # 每控制步 mean(|tau_exo|)
                'exo_sq_sum':  0.0,       # 每控制步 sum(tau_exo^2)
                'exo_abs_max': 0.0,       # max(|tau_exo|)
                # 回合级
                'step_count':  0,         # steps_in_episode（控制步数）
                'ret_sum':     0.0        # episode_return（控制步逐步累加 reward）
            }
        self.ep_stats = [_new_ep_stat() for _ in range(self.num_slaves)]

        # —— CSV 初始化（两份：baseline 与 exo 扩展） —— #
        self.csv_baseline = os.path.join('..','nn','episode_metrics_baseline.csv')
        self.csv_exo      = os.path.join('..','nn','episode_metrics.csv')
        _ensure_dir(self.csv_baseline); _ensure_dir(self.csv_exo)
        if not os.path.exists(self.csv_baseline):
            with open(self.csv_baseline,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'eval_round','slave_id',
                    'avg_tau_abs','max_tau_abs','sum_tau_sq',
                    'avg_activation','max_activation','sum_activation_sq',
                    'episode_return','steps_in_episode',
                    'ctrl_avg_power','ctrl_energy'
                ])
        if not os.path.exists(self.csv_exo):
            with open(self.csv_exo,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'eval_round','slave_id',
                    # 与 baseline 一致的部分（便于合并分析）
                    'avg_tau_abs','max_tau_abs','sum_tau_sq',
                    'avg_activation','max_activation','sum_activation_sq',
                    'episode_return','steps_in_episode',
                    # 额外：策略噪声
                    'noise_pd','noise_exo',
                    # 外骨骼：力矩统计（Python 侧统计）
                    'exo_tau_avg_abs','exo_tau_max_abs','exo_tau_sum_sq',
                    # 外骨骼：功率/能量（C++ 侧逐回合统计）
                    'exo_avg_power','exo_energy',
                    # 控制功率/能量（C++ 侧逐回合统计）
                    'ctrl_avg_power','ctrl_energy'
                ])

        # —— 让 C++ 环境初始化 —— #
        self.env.Resets(True)

    # ---------- 保存/加载 ----------
    def SaveModel(self):
        self.model.save('../nn/current.pt')
        self.exo_model.save('../nn/current_exo.pt')
        self.muscle_model.save('../nn/current_muscle.pt')
        if self.max_return_epoch == self.num_evaluation:
            self.model.save('../nn/max.pt')
            self.exo_model.save('../nn/max_exo.pt')
            self.muscle_model.save('../nn/max_muscle.pt')
        if self.num_evaluation%100 == 0:
            k = self.num_evaluation//100
            self.model.save(f'../nn/{k}.pt')
            self.exo_model.save(f'../nn/{k}_exo.pt')
            self.muscle_model.save(f'../nn/{k}_muscle.pt')

    def LoadModel(self,path):
        self.model.load(f'../nn/{path}.pt')
        self.exo_model.load(f'../nn/{path}_exo.pt')
        self.muscle_model.load(f'../nn/{path}_muscle.pt')

    # ---------- GAE 计算 ----------
    def ComputeTDandGAE(self):
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0

        for epi in self.total_episodes:
            data = epi.GetData()
            n = len(data)
            if n == 0: 
                continue
            states, actions, rewards, values, logprobs = zip(*data)
            values = np.concatenate((values, np.zeros(1)), axis=0)
            adv = np.zeros(n); ad_t = 0.0

            epi_return = 0.0
            for i in reversed(range(n)):
                epi_return += rewards[i]
                delta = rewards[i] + values[i+1]*self.gamma - values[i]
                ad_t  = delta + self.gamma*self.lb*ad_t
                adv[i]= ad_t
            self.sum_return += epi_return
            TD = values[:n] + adv
            for i in range(n):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], adv[i])

        self.num_episode = len(self.total_episodes)
        self.num_tuple   = len(self.replay_buffer.buffer)
        print('SIM : {}'.format(self.num_tuple))
        self.num_tuple_so_far += self.num_tuple

        # 肌肉自监督数据
        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA']    = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L']      = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b']      = self.env.GetMuscleTuplesb()

    # ---------- 采样 +（新增）逐回合统计 ----------
    def GenerateTransitions(self):
        self.total_episodes = []
        states = self.env.GetStates()
        local_step = 0
        counter = 0

        # 为防止 CSV 频繁写入，这一轮评估结束要落盘的回合缓存
        ep_rows_baseline = []   # [(eval_round, slave_id, ...), ...]
        ep_rows_exo      = []

        # 小工具：回合结束时写一行到缓存，并重置该 slave 的统计
        def _flush_episode_row(slave_id, eval_round, noise_pd, noise_exo):
            st = self.ep_stats[slave_id]
            steps = max(1, st['step_count'])  # 防 0

            # ===== baseline 列（你要对齐那 11 个） + 控制功率/能量 =====
            avg_tau_abs = st['tau_abs_sum']/steps
            max_tau_abs = st['tau_abs_max']
            sum_tau_sq  = st['tau_sq_sum']
            avg_act_abs = st['act_abs_sum']/steps if self.use_muscle else float('nan')
            max_act_abs = st['act_abs_max']   if self.use_muscle else float('nan')
            sum_act_sq  = st['act_sq_sum']    if self.use_muscle else 0.0

            # —— 与无外骨骼版一致：从 C++ 读回合功率/能量 —— #
            try:
                ctrl_avg_power_vec = self.env.GetEpisodeCtrlAvgPowerVec()
                ctrl_energy_vec    = self.env.GetEpisodeCtrlEnergyVec()
                ctrl_avg_power = float(ctrl_avg_power_vec[slave_id])
                ctrl_energy    = float(ctrl_energy_vec[slave_id])
            except:
                ctrl_avg_power = float('nan')
                ctrl_energy    = float('nan')

            row_base = [
                eval_round, slave_id,
                avg_tau_abs, max_tau_abs, sum_tau_sq,
                avg_act_abs, max_act_abs, sum_act_sq,
                st['ret_sum'], st['step_count'],
                ctrl_avg_power, ctrl_energy
            ]
            ep_rows_baseline.append(row_base)

            # ===== exo 扩展列（外骨骼力矩 Python 统计；功率/能量从 C++ 读向量）=====
            exo_avg_abs = st['exo_abs_sum']/steps
            exo_max_abs = st['exo_abs_max']
            exo_sum_sq  = st['exo_sq_sum']
            try:
                exo_avg_power_vec = self.env.GetEpisodeExoAvgPowerVec()
                exo_energy_vec    = self.env.GetEpisodeExoEnergyVec()
                exo_avg_power = float(exo_avg_power_vec[slave_id])
                exo_energy    = float(exo_energy_vec[slave_id])
            except:
                exo_avg_power = float('nan')
                exo_energy    = float('nan')

            row_exo = [
                eval_round, slave_id,
                avg_tau_abs, max_tau_abs, sum_tau_sq,
                avg_act_abs, max_act_abs, sum_act_sq,
                st['ret_sum'], st['step_count'],
                noise_pd, noise_exo,
                exo_avg_abs, exo_max_abs, exo_sum_sq,
                exo_avg_power, exo_energy,
                ctrl_avg_power, ctrl_energy
            ]
            ep_rows_exo.append(row_exo)

            # 重置该 slave 的统计，开始下一回合
            self.ep_stats[slave_id] = {
                k:(0.0 if isinstance(v,float) else 0) for k,v in self.ep_stats[slave_id].items()
            }

        while True:
            counter += 1
            if counter % 10 == 0:
                print('SIM : {}'.format(local_step), end='\r')

            # 两个策略头：PD + Exo
            a_dist_pd, v = self.model(Tensor(states))
            a_dist_exo   = self.exo_model(Tensor(states))

            a_pd  = a_dist_pd.sample()   # [N, A]
            a_exo = a_dist_exo.sample()  # [N, A]

            a_combined = torch.cat([a_pd, a_exo], dim=1).cpu().detach().numpy()
            logprob_pd  = a_dist_pd.log_prob(a_pd).cpu().detach().numpy().reshape(-1)
            logprob_exo = a_dist_exo.log_prob(a_exo).cpu().detach().numpy().reshape(-1)
            logprobs    = (logprob_pd + logprob_exo)
            values      = v.cpu().detach().numpy().reshape(-1)

            # 下发到环境
            self.env.SetActions(a_pd.cpu().detach().numpy())
            self.env.SetExoTorques(a_exo.cpu().detach().numpy())

            # ===== 力矩统计（控制步：对 τ_des 只在控制步更新一次）
            # 我们在控制步内取 τ_des 的“瞬时值”，并将该步代表值记入回合累计
            dt_np = self.env.GetDesiredTorques()     # shape: [N, A]
            tau_abs = np.abs(dt_np)
            tau_sq  = dt_np*dt_np
            mean_abs_tau = tau_abs.mean(axis=1)      # [N]  —— 本控制步的 mean(|tau|)
            max_abs_tau  = tau_abs.max(axis=1)       # [N]  —— 本控制步内“瞬时最大”；若需要细步峰值，请在肌肉细步聚合里更新
            sum_tau_sq   = tau_sq.sum(axis=1)        # [N]  —— 本控制步的 sum(tau^2)

            # 外骨骼力矩统计（Python 侧；扩展 CSV 用）
            exo_np = a_exo.cpu().detach().numpy()    # [N, A]
            exo_abs = np.abs(exo_np)
            exo_sq  = exo_np*exo_np

            for j in range(self.num_slaves):
                st = self.ep_stats[j]
                st['tau_abs_sum'] += float(mean_abs_tau[j])
                st['tau_sq_sum']  += float(sum_tau_sq[j])
                # —— max_tau_abs：默认采用“控制步内的细步峰值”。这里先用瞬时值占位，细步循环里还会用 max() 扩充。
                st['tau_abs_max']  = max(st['tau_abs_max'], float(max_abs_tau[j]))
                st['exo_abs_sum'] += float(exo_abs[j].mean())
                st['exo_sq_sum']  += float(exo_sq[j].sum())
                st['exo_abs_max']  = max(st['exo_abs_max'], float(exo_abs[j].max()))

            # ===== 肌肉细步（用来：驱动环境积分 + 聚合激活；并可提升 max_tau_abs 为细步峰值）=====
            if self.use_muscle:
                mt = Tensor(self.env.GetMuscleTorques())
                for _ in range(self.num_sim_per_control//2):
                    dt = Tensor(self.env.GetDesiredTorques())
                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()  # [N, M]
                    self.env.SetActivationLevels(activations)

                    # —— 细步统计（聚合到控制步级，再进入回合累计）——
                    act_abs = np.abs(activations)
                    act_sq  = activations*activations
                    mean_abs_act = act_abs.mean(axis=1)   # [N]
                    max_abs_act  = act_abs.max(axis=1)
                    sum_act_sq   = act_sq.sum(axis=1)

                    # 细步里也用 τ_des 的瞬时值更新“控制步内峰值”，从而让 max_tau_abs 捕捉细步尖峰
                    dt_np_sub = dt.cpu().detach().numpy()
                    tau_abs_sub_max = np.abs(dt_np_sub).max(axis=1)  # [N]

                    for j in range(self.num_slaves):
                        st = self.ep_stats[j]
                        st['act_abs_sum'] += float(mean_abs_act[j])   # 控制步层面最终会除以 step_count
                        st['act_sq_sum']  += float(sum_act_sq[j])     # 累加求和，不做平均
                        st['act_abs_max']  = max(st['act_abs_max'], float(max_abs_act[j]))
                        # —— 提升 max_tau_abs 为控制步内的“细步峰值” —— #
                        st['tau_abs_max']  = max(st['tau_abs_max'], float(tau_abs_sub_max[j]))

                    self.env.Steps(2)
            else:
                self.env.StepsAtOnce()

            # ===== 奖励与终止处理（控制步级）=====
            try:
                noise_pd  = self.model.log_std.exp().mean().item()
            except:
                noise_pd  = float('nan')
            try:
                noise_exo = self.exo_model.log_std.exp().mean().item()
            except:
                noise_exo = float('nan')

            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if (np.any(np.isnan(states[j])) or np.any(np.isnan(a_combined[j])) or
                    np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j]))):
                    nan_occur = True
                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    r = self.env.GetReward(j)         # —— episode_return：控制步逐步累加 —— #
                    self.episodes[j].Push(states[j], a_combined[j], r, values[j], logprobs[j])

                    st = self.ep_stats[j]
                    st['ret_sum']   += float(r)
                    st['step_count']+= 1              # —— steps_in_episode：控制步计数 —— #
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    # —— 回合完成：写两份 CSV（缓存）；eval_round = num_evaluation + 1 —— #
                    _flush_episode_row(j, self.num_evaluation+1, noise_pd, noise_exo)

                    # 结束本回合：存入 PPO episode 缓冲并开启新回合
                    self.total_episodes.append(self.episodes[j])
                    self.episodes[j] = EpisodeBuffer()
                    # 重置 C++ 环境回合
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

        # —— 把本轮所有回合写盘 —— #
        if ep_rows_baseline:
            with open(self.csv_baseline,'a',newline='') as f:
                w = csv.writer(f)
                w.writerows(ep_rows_baseline)
        if ep_rows_exo:
            with open(self.csv_exo,'a',newline='') as f:
                w = csv.writer(f)
                w.writerows(ep_rows_exo)

    # ---------- 优化（PPO + 肌肉） ----------
    def OptimizeSimulationNN(self):
        all_trans = np.array(self.replay_buffer.buffer)
        for e in range(self.num_epochs):
            np.random.shuffle(all_trans)
            for i in range(len(all_trans)//self.batch_size):
                transitions = all_trans[i*self.batch_size:(i+1)*self.batch_size]
                batch = Transition(*zip(*transitions))
                stack_s  = np.vstack(batch.s).astype(np.float32)
                stack_a  = np.vstack(batch.a).astype(np.float32) # [B, 2*A]
                stack_lp = np.vstack(batch.logprob).astype(np.float32)
                stack_td = np.vstack(batch.TD).astype(np.float32)
                stack_gae= np.vstack(batch.GAE).astype(np.float32)

                a_pd   = Tensor(stack_a[:, :self.num_action])
                a_exo  = Tensor(stack_a[:, self.num_action:])

                a_dist_pd, v = self.model(Tensor(stack_s))
                a_dist_exo   = self.exo_model(Tensor(stack_s))
                logprob_now  = a_dist_pd.log_prob(a_pd) + a_dist_exo.log_prob(a_exo)
                ratio        = torch.exp(logprob_now - Tensor(stack_lp))

                stack_gae = (stack_gae - stack_gae.mean())/(stack_gae.std()+1e-5)
                stack_gae = Tensor(stack_gae)
                stack_td  = Tensor(stack_td)

                loss_critic = ((v - stack_td).pow(2)).mean()
                surrogate1  = ratio * stack_gae
                surrogate2  = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * stack_gae
                loss_actor  = - torch.min(surrogate1, surrogate2).mean()
                loss_entropy= - self.w_entropy*(a_dist_pd.entropy().mean() + a_dist_exo.entropy().mean())

                self.loss_actor  = loss_actor.cpu().detach().numpy().tolist()
                self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                loss = loss_actor + loss_entropy + loss_critic
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for p in list(self.model.parameters())+list(self.exo_model.parameters()):
                    if p.grad is not None:
                        p.grad.data.clamp_(-0.5,0.5)
                self.optimizer.step()
            print(f'Optimizing sim/exo nn : {e+1}/{self.num_epochs}', end='\r')
        print('')

    def _generate_shuffle_indices(self, n, m):
        p = np.random.permutation(n)
        r = m - n%m
        if r>0: p = np.hstack([p, np.random.randint(0,n,r)])
        return p.reshape(-1,m)

    def OptimizeMuscleNN(self):
        for e in range(self.num_epochs_muscle):
            mb = self._generate_shuffle_indices(self.muscle_buffer['JtA'].shape[0], self.muscle_batch_size)
            for idx in mb:
                JtA     = Tensor(self.muscle_buffer['JtA'][idx].astype(np.float32))
                tau_des = Tensor(self.muscle_buffer['TauDes'][idx].astype(np.float32))
                L       = Tensor(self.muscle_buffer['L'][idx].astype(np.float32))
                L       = L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)
                b       = Tensor(self.muscle_buffer['b'][idx].astype(np.float32))

                activation = self.muscle_model(JtA, tau_des)
                tau = torch.einsum('ijk,ik->ij', (L, activation)) + b

                loss_reg    = (activation).pow(2).mean()
                loss_target = (((tau - tau_des)/100.0).pow(2)).mean()
                loss = 0.01*loss_reg + loss_target

                self.optimizer_muscle.zero_grad()
                loss.backward(retain_graph=True)
                for p in self.muscle_model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-0.5,0.5)
                self.optimizer_muscle.step()
            print(f'Optimizing muscle nn : {e+1}/{self.num_epochs_muscle}', end='\r')
        self.loss_muscle = loss.cpu().detach().numpy().tolist()
        print('')

    def OptimizeModel(self):
        self.ComputeTDandGAE()
        self.OptimizeSimulationNN()
        if self.use_muscle:
            self.OptimizeMuscleNN()

    def Train(self):
        self.GenerateTransitions()
        self.OptimizeModel()

    def Evaluate(self):
        self.num_evaluation = self.num_evaluation + 1
        elapsed = time.time() - self.tic
        h = int(elapsed//3600.0); m = int(elapsed//60.0) - h*60; s = int(elapsed) - h*3600 - m*60
        if self.num_episode == 0: self.num_episode = 1
        if self.num_tuple   == 0: self.num_tuple   = 1
        avg_return = self.sum_return/self.num_episode
        if self.max_return < avg_return:
            self.max_return = avg_return
            self.max_return_epoch = self.num_evaluation

        try: noise_pd  = self.model.log_std.exp().mean().item()
        except: noise_pd = float('nan')
        try: noise_exo = self.exo_model.log_std.exp().mean().item()
        except: noise_exo = float('nan')

        print(f'# {self.num_evaluation} === {h}h:{m}m:{s}s ===')
        print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
        print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
        print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
        print('||Noise (PD)               : {:.3f}'.format(noise_pd))
        print('||Noise (Exo)              : {:.3f}'.format(noise_exo))
        print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
        print('||Num Transition           : {}'.format(self.num_tuple))
        print('||Num Episode              : {}'.format(self.num_episode))
        print('||Avg Return per episode   : {:.3f}'.format(avg_return))
        print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
        print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
        print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
        self.rewards.append(avg_return)

        self.SaveModel()
        print('=============================================')
        return np.array(self.rewards)

# ===== 可视化 =====
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Plot(y,title,num_fig=1,ylim=True):
    temp_y = np.zeros(y.shape)
    if y.shape[0]>5:
        temp_y[0] = y[0]
        temp_y[1] = 0.5*(y[0] + y[1])
        temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
        temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
        for i in range(4,y.shape[0]):
            temp_y[i] = np.sum(y[i-4:i+1])*0.2
    plt.figure(num_fig); plt.clf(); plt.title(title)
    plt.plot(y,'b'); plt.plot(temp_y,'r')
    if ylim: plt.ylim([0,1])
    plt.show(); plt.pause(0.001)
    if y.shape[0] % 100 == 0:
        save_dir = os.path.join('..','nn','pics'); os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{title}_{y.shape[0]}.png"))

# ===== 打到文件的 tee =====
class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message); self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()

# ===== 主入口 =====
import argparse
if __name__=="__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(f"train_log_{timestamp}.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',help='model path')
    parser.add_argument('-d','--meta', help='meta file')
    args = parser.parse_args()
    if args.meta is None:
        print('Provide meta file'); exit()

    ppo = PPO(args.meta)
    nn_dir = '../nn/pics'
    if not os.path.exists(nn_dir): os.makedirs(nn_dir)
    if args.model is not None: ppo.LoadModel(args.model)
    else: ppo.SaveModel()

    print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(), ppo.env.GetNumAction()))
    for _ in range(ppo.max_iteration-5):
        ppo.Train()
        rewards = ppo.Evaluate()
        Plot(rewards, 'reward', 0, False)
