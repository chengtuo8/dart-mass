# -*- coding: utf-8 -*-
# 无外骨骼版 main.py（对齐补丁版）
# 口径约定：
#   1) 策略相关（avg_tau_abs / sum_tau_sq / avg_activation / steps_in_episode）按【控制步】计数与平均
#   2) max_tau_abs 在控制步内按细步聚合一次（捕捉该控制步内的峰值），回合取全局最大
#      —— 如需严格“控制步瞬时”而不看细步峰值，可按下方注释切换
#   3) ctrl_avg_power / ctrl_energy 统一从 C++ 回合向量读取：GetEpisodeCtrlAvgPowerVec / GetEpisodeCtrlEnergyVec
#   4) eval_round 写入 num_evaluation + 1，与外骨骼版一致

import math
import random
import time
import csv
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import pymss  # type: ignore
from Model import *
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob'))
class EpisodeBuffer(object):
    def __init__(self):
        self.data = []
    def Push(self, *args):
        self.data.append(Episode(*args))
    def Pop(self):
        self.data.pop()
    def GetData(self):
        return self.data

MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))
class MuscleBuffer(object):
    def __init__(self, buff_size=10000):
        super(MuscleBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)
    def Push(self, *args):
        self.buffer.append(MuscleTransition(*args))
    def Clear(self):
        self.buffer.clear()

Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
    def __init__(self, buff_size=10000):
        super(ReplayBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)
    def Push(self, *args):
        self.buffer.append(Transition(*args))
    def Clear(self):
        self.buffer.clear()

class PPO(object):
    def __init__(self, meta_file):
        np.random.seed(seed=int(time.time()))
        self.num_slaves = 16
        self.env = pymss.pymss(meta_file, self.num_slaves)
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()

        self.num_epochs = 10
        self.num_epochs_muscle = 3
        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.num_episode = 0
        self.num_tuple = 0
        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.gamma = 0.99
        self.lb = 0.99

        self.buffer_size = 2048
        self.batch_size = 128
        self.muscle_batch_size = 128
        self.replay_buffer = ReplayBuffer(30000)
        self.muscle_buffer = {}

        self.model = SimulationNN(self.num_state, self.num_action)
        self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(), self.num_action, self.num_muscles)
        if use_cuda:
            self.model.cuda()
            self.muscle_model.cuda()

        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        self.max_iteration = 50000

        self.w_entropy = -0.001

        self.loss_actor = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards = []
        self.sum_return = 0.0
        self.max_return = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()

        self.episodes = [EpisodeBuffer() for _ in range(self.num_slaves)]
        self.env.Resets(True)

        # ====== 每回合指标 CSV（固定列）======
        self.metrics_path = "../nn/episode_metrics.csv"
        self._metrics_header_written = os.path.exists(self.metrics_path)

        # 回合累计器（按【控制步】统计）
        self._alloc_episode_accumulators()

    # ----------------------------------------------------------
    # 回合累计器：全在【控制步】层面更新一次
    # ----------------------------------------------------------
    def _alloc_episode_accumulators(self):
        n = self.num_slaves
        self._epi_step_count  = [0]    * n  # steps_in_episode（控制步数）
        self._epi_return_sum  = [0.0]  * n  # episode_return（控制步逐步累加 reward）
        # 力矩相关（控制步聚合一次）
        self._epi_tau_abs_sum = [0.0]  * n  # 累加“每个控制步的 mean(|tau|)”，最后 / 步数 → avg_tau_abs
        self._epi_tau_abs_max = [0.0]  * n  # 回合内全局 max(|tau|)
        self._epi_tau_sq_sum  = [0.0]  * n  # 累加“每个控制步内的 sum(tau^2)”
        # 激活度相关（肌肉模式有效；控制步聚合一次）
        if self.use_muscle:
            self._epi_act_mean_sum = [0.0] * n  # 累加“每个控制步的 mean(a)”，最后 / 步数 → avg_activation
            self._epi_act_max      = [0.0] * n  # 回合内全局 max(a)
            self._epi_act_sq_sum   = [0.0] * n  # 累加“每个控制步内（跨细步+跨肌肉）的 a^2 总和” → sum_activation_sq

    # ----------------------------------------------------------
    # 写一行 CSV（回合结束时）
    #   ctrl_avg_power / ctrl_energy：从 C++ 回合向量读取
    #   eval_round：num_evaluation + 1
    # ----------------------------------------------------------
    def _write_episode_metrics_row(self, j):
        sc = max(1, self._epi_step_count[j])  # 防除零
        avg_tau_abs = self._epi_tau_abs_sum[j] / sc
        max_tau_abs = self._epi_tau_abs_max[j]
        sum_tau_sq  = self._epi_tau_sq_sum[j]
        episode_return = self._epi_return_sum[j]
        steps_in_episode = sc

        if self.use_muscle:
            avg_activation    = self._epi_act_mean_sum[j] / sc
            max_activation    = self._epi_act_max[j]
            sum_activation_sq = self._epi_act_sq_sum[j]
        else:
            # 无肌肉：按外骨骼版口径写占位（可选 NaN/空）
            avg_activation    = float('nan')
            max_activation    = float('nan')
            sum_activation_sq = 0.0

        # —— 与外骨骼版对齐：从 C++ 端读取回合统计向量（与 substep 无关）——
        try:
            p_vec = np.asarray(self.env.GetEpisodeCtrlAvgPowerVec(), dtype=np.float64).ravel()
            e_vec = np.asarray(self.env.GetEpisodeCtrlEnergyVec(),   dtype=np.float64).ravel()
            ctrl_avg_power = float(p_vec[j]) if j < len(p_vec) else 0.0
            ctrl_energy    = float(e_vec[j]) if j < len(e_vec) else 0.0
        except Exception:
            # 若接口暂不可用，回退为 0，并在日志提示（不改变表头）
            ctrl_avg_power = 0.0
            ctrl_energy    = 0.0
            print("[warn] C++ ctrl power/energy vectors not available; writing 0.0")

        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        need_header = not self._metrics_header_written
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow([
                    'eval_round','slave_id',
                    'avg_tau_abs','max_tau_abs','sum_tau_sq',
                    'avg_activation','max_activation','sum_activation_sq',
                    'episode_return','steps_in_episode',
                    'ctrl_avg_power','ctrl_energy'
                ])
                self._metrics_header_written = True
            # eval_round 采用 num_evaluation + 1，与外骨骼版一致
            w.writerow([
                self.num_evaluation + 1, j,
                f"{avg_tau_abs:.6f}", f"{max_tau_abs:.6f}", f"{sum_tau_sq:.6f}",
                f"{avg_activation:.6f}", f"{max_activation:.6f}", f"{sum_activation_sq:.6f}",
                f"{episode_return:.6f}", steps_in_episode,
                f"{ctrl_avg_power:.6f}", f"{ctrl_energy:.6f}"
            ])

        # 写完当前 j 的回合指标后，清零其累计器（便于该 env 下个回合继续跑）
        self._epi_step_count[j]  = 0
        self._epi_return_sum[j]  = 0.0
        self._epi_tau_abs_sum[j] = 0.0
        self._epi_tau_abs_max[j] = 0.0
        self._epi_tau_sq_sum[j]  = 0.0
        if self.use_muscle:
            self._epi_act_mean_sum[j] = 0.0
            self._epi_act_max[j]      = 0.0
            self._epi_act_sq_sum[j]   = 0.0

    def SaveModel(self):
        self.model.save('../nn/current.pt')
        self.muscle_model.save('../nn/current_muscle.pt')
        if self.max_return_epoch == self.num_evaluation:
            self.model.save('../nn/max.pt')
            self.muscle_model.save('../nn/max_muscle.pt')
        if self.num_evaluation % 100 == 0:
            self.model.save('../nn/' + str(self.num_evaluation // 100) + '.pt')
            self.muscle_model.save('../nn/' + str(self.num_evaluation // 100) + '_muscle.pt')

    def LoadModel(self, path):
        self.model.load('../nn/' + path + '.pt')
        self.muscle_model.load('../nn/' + path + '_muscle.pt')

    def ComputeTDandGAE(self):
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0
        for epi in self.total_episodes:
            data = epi.GetData()
            size = len(data)
            if size == 0:
                continue
            states, actions, rewards, values, logprobs = zip(*data)
            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0
            epi_return = 0.0
            for i in reversed(range(len(data))):
                epi_return += rewards[i]
                delta = rewards[i] + values[i+1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t
            self.sum_return += epi_return
            TD = values[:size] + advantages
            for i in range(size):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
        self.num_episode = len(self.total_episodes)
        self.num_tuple = len(self.replay_buffer.buffer)
        print('SIM : {}'.format(self.num_tuple))
        self.num_tuple_so_far += self.num_tuple

        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA'] = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L'] = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b'] = self.env.GetMuscleTuplesb()
        print(self.muscle_buffer['JtA'].shape)

    def GenerateTransitions(self):
        self.total_episodes = []
        states = [None]*self.num_slaves
        actions = [None]*self.num_slaves
        rewards = [None]*self.num_slaves
        states_next = [None]*self.num_slaves
        states = self.env.GetStates()
        local_step = 0
        terminated = [False]*self.num_slaves
        counter = 0

        # 开始新一轮采样前，清空累计器
        self._alloc_episode_accumulators()

        while True:
            counter += 1
            if counter % 10 == 0:
                print('SIM : {}'.format(local_step), end='\r')

            a_dist, v = self.model(Tensor(states))
            actions = a_dist.sample().cpu().detach().numpy()
            logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
            values = v.cpu().detach().numpy().reshape(-1)
            self.env.SetActions(actions)

            if self.use_muscle:
                mt = Tensor(self.env.GetMuscleTorques())

                # ===== 控制步开始：细步内先做【控制步级】聚合，再在控制步末尾只记一次 =====
                # 以下是“本控制步”的临时统计量（会合并细步信息）
                ctrl_tau_abs_mean_sum = np.zeros(self.num_slaves, dtype=np.float64)  # 跨细步累加 mean(|tau|)，控制步末/num_substep
                ctrl_tau_sq_sum       = np.zeros(self.num_slaves, dtype=np.float64)  # 跨细步累加 sum(tau^2)
                ctrl_tau_abs_max      = np.zeros(self.num_slaves, dtype=np.float64)  # 该控制步内峰值 |tau|
                ctrl_act_mean_sum     = np.zeros(self.num_slaves, dtype=np.float64)  # 跨细步累加 mean(a)，控制步末/num_substep
                ctrl_act_sq_sum       = np.zeros(self.num_slaves, dtype=np.float64)  # 跨细步 + 跨肌肉 累加 a^2
                ctrl_act_max          = np.zeros(self.num_slaves, dtype=np.float64)  # 该控制步内峰值 a
                substep_counter       = 0

                for _ in range(self.num_simulation_per_control // 2):
                    # 细步：取“期望力矩”和“肌肉激活”，但不在细步层面直接计入回合
                    dt = Tensor(self.env.GetDesiredTorques())
                    dt_np = dt.cpu().detach().numpy()
                    # 细步层面先聚合到“控制步临时量”
                    ctrl_tau_abs_mean_sum += np.abs(dt_np).mean(axis=1)        # 每细步的 mean(|tau|)
                    ctrl_tau_sq_sum       += (dt_np**2).sum(axis=1)            # 每细步的 sum(tau^2)
                    ctrl_tau_abs_max       = np.maximum(ctrl_tau_abs_max, np.abs(dt_np).max(axis=1))  # 该细步的峰值并入本控制步峰值
                    substep_counter += 1

                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                    ctrl_act_mean_sum += np.abs(activations).mean(axis=1)      # 每细步 mean(a)（取 abs 可选；若不取 abs 删掉 np.abs）
                    ctrl_act_sq_sum   += (activations**2).sum(axis=1)          # 每细步 sum(a^2)
                    ctrl_act_max       = np.maximum(ctrl_act_max, np.abs(activations).max(axis=1))
                    self.env.SetActivationLevels(activations)
                    self.env.Steps(2)

                # —— 控制步结束：把上面的细步信息汇总为一次“控制步级”更新（真正进入回合累计）——
                # 1) 力矩
                ctrl_tau_abs_mean = ctrl_tau_abs_mean_sum / max(1, substep_counter)  # 该控制步代表性的 mean(|tau|)
                for j in range(self.num_slaves):
                    self._epi_tau_abs_sum[j] += float(ctrl_tau_abs_mean[j])
                    self._epi_tau_sq_sum[j]  += float(ctrl_tau_sq_sum[j])
                    # max_tau_abs：回合内全局最大；这里采用“控制步内细步峰值”，更容易捕捉尖峰
                    # 如需严格使用“控制步瞬时值（不看细步）”，可把下面这一行替换为：
                    #   self._epi_tau_abs_max[j] = max(self._epi_tau_abs_max[j], float(np.abs(dt_np[j]).max()))
                    self._epi_tau_abs_max[j]  = max(self._epi_tau_abs_max[j], float(ctrl_tau_abs_max[j]))

                # 2) 激活度（仅肌肉模式有效）
                ctrl_act_mean = ctrl_act_mean_sum / max(1, substep_counter)    # 该控制步代表性的 mean(a)
                for j in range(self.num_slaves):
                    self._epi_act_mean_sum[j] += float(ctrl_act_mean[j])
                    self._epi_act_sq_sum[j]   += float(ctrl_act_sq_sum[j])
                    self._epi_act_max[j]       = max(self._epi_act_max[j], float(ctrl_act_max[j]))

                # 3) 控制步计数（steps_in_episode 以控制步为单位）
                for j in range(self.num_slaves):
                    self._epi_step_count[j] += 1

            else:
                # 非肌肉模式：同样按【控制步】统计一次
                dt = Tensor(self.env.GetDesiredTorques())
                dt_np = dt.cpu().detach().numpy()
                # 这里没有细步循环，直接将本控制步的瞬时值作为该控制步代表
                for j in range(self.num_slaves):
                    self._epi_tau_abs_sum[j] += float(np.abs(dt_np[j]).mean())
                    self._epi_tau_sq_sum[j]  += float((dt_np[j]**2).sum())
                    self._epi_tau_abs_max[j]  = max(self._epi_tau_abs_max[j], float(np.abs(dt_np[j]).max()))
                    self._epi_step_count[j]  += 1
                self.env.StepsAtOnce()

            # ====== 奖励与回合终止处理（控制步级）======
            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if (np.any(np.isnan(states[j])) or
                    np.any(np.isnan(actions[j])) or
                    np.any(np.isnan(values[j])) or
                    np.any(np.isnan(logprobs[j]))):
                    nan_occur = True

                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    rewards[j] = self.env.GetReward(j)  # 每个【控制步】记一次奖励
                    self._epi_return_sum[j] += float(rewards[j])
                    self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    self.total_episodes.append(self.episodes[j])

                    # —— 回合结束：写 CSV 一行（用 C++ 回合功率/能量；eval_round=+1）——
                    self._write_episode_metrics_row(j)

                    self.episodes[j] = EpisodeBuffer()
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

    def OptimizeSimulationNN(self):
        all_transitions = np.array(self.replay_buffer.buffer)
        for j in range(self.num_epochs):
            np.random.shuffle(all_transitions)
            for i in range(len(all_transitions)//self.batch_size):
                transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
                batch = Transition(*zip(*transitions))
                stack_s = np.vstack(batch.s).astype(np.float32)
                stack_a = np.vstack(batch.a).astype(np.float32)
                stack_lp = np.vstack(batch.logprob).astype(np.float32)
                stack_td = np.vstack(batch.TD).astype(np.float32)
                stack_gae = np.vstack(batch.GAE).astype(np.float32)

                a_dist, v = self.model(Tensor(stack_s))
                # Critic
                loss_critic = ((v - Tensor(stack_td)).pow(2)).mean()
                # Actor (PPO clip)
                ratio = torch.exp(a_dist.log_prob(Tensor(stack_a)) - Tensor(stack_lp))
                stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1E-5)
                stack_gae = Tensor(stack_gae)
                surrogate1 = ratio * stack_gae
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * stack_gae
                loss_actor = - torch.min(surrogate1, surrogate2).mean()
                # Entropy
                loss_entropy = - self.w_entropy * a_dist.entropy().mean()

                self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
                self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                loss = loss_actor + loss_entropy + loss_critic
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
            print('Optimizing sim nn : {}/{}'.format(j+1, self.num_epochs), end='\r')
        print('')

    def generate_shuffle_indices(self, batch_size, minibatch_size):
        n = batch_size
        m = minibatch_size
        p = np.random.permutation(n)
        r = m - n % m
        if r > 0:
            p = np.hstack([p, np.random.randint(0, n, r)])
        p = p.reshape(-1, m)
        return p

    def OptimizeMuscleNN(self):
        for j in range(self.num_epochs_muscle):
            minibatches = self.generate_shuffle_indices(self.muscle_buffer['JtA'].shape[0], self.muscle_batch_size)
            for minibatch in minibatches:
                stack_JtA = self.muscle_buffer['JtA'][minibatch].astype(np.float32)
                stack_tau_des = self.muscle_buffer['TauDes'][minibatch].astype(np.float32)
                stack_L = self.muscle_buffer['L'][minibatch].astype(np.float32)
                stack_L = stack_L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)
                stack_b = self.muscle_buffer['b'][minibatch].astype(np.float32)

                stack_JtA = Tensor(stack_JtA)
                stack_tau_des = Tensor(stack_tau_des)
                stack_L = Tensor(stack_L)
                stack_b = Tensor(stack_b)

                activation = self.muscle_model(stack_JtA, stack_tau_des)
                tau = torch.einsum('ijk,ik->ij', (stack_L, activation)) + stack_b

                loss_reg = (activation).pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target
                self.optimizer_muscle.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.muscle_model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer_muscle.step()
            print('Optimizing muscle nn : {}/{}'.format(j+1, self.num_epochs_muscle), end='\r')
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
        h = int((time.time() - self.tic)//3600.0)
        m = int((time.time() - self.tic)//60.0)
        s = int((time.time() - self.tic))
        m = m - h*60
        s = int((time.time() - self.tic))
        s = s - h*3600 - m*60
        if self.num_episode == 0:
            self.num_episode = 1
        if self.num_tuple == 0:
            self.num_tuple = 1
        if self.max_return < self.sum_return/self.num_episode:
            self.max_return = self.sum_return/self.num_episode
            self.max_return_epoch = self.num_evaluation
        print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation, h, m, s))
        print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
        print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
        print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
        print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))
        print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
        print('||Num Transition           : {}'.format(self.num_tuple))
        print('||Num Episode              : {}'.format(self.num_episode))
        print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
        print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
        print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
        print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return, self.max_return_epoch))
        self.rewards.append(self.sum_return/self.num_episode)
        self.SaveModel()
        print('=============================================')
        return np.array(self.rewards)

import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Plot(y, title, num_fig=1, ylim=True):
    temp_y = np.zeros(y.shape)
    if y.shape[0] > 5:
        temp_y[0] = y[0]
        temp_y[1] = 0.5*(y[0] + y[1])
        temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
        temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
        for i in range(4, y.shape[0]):
            temp_y[i] = np.sum(y[i-4:i+1]) * 0.2

    plt.figure(num_fig)
    plt.clf()
    plt.title(title)
    plt.plot(y, 'b')
    plt.plot(temp_y, 'r')
    plt.show()
    if ylim:
        plt.ylim([0, 1])
    plt.pause(0.001)

    # —— 可选：每隔100次评估保存一次图 —— 
    eval_count = y.shape[0]
    if eval_count % 100 == 0:
        save_dir = os.path.join('..', 'nn', 'pics')
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{title}_{eval_count}.png")
        plt.savefig(filename)
        print(f"[Plot] Saved figure to {filename}")

class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

import argparse
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(f"train_log_{timestamp}.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model path')
    parser.add_argument('-d', '--meta', help='meta file')
    args = parser.parse_args()
    if args.meta is None:
        print('Provide meta file')
        exit()

    ppo = PPO(args.meta)
    nn_dir = '../nn/pics'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()
    print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(), ppo.env.GetNumAction()))
    for i in range(ppo.max_iteration-5):
        ppo.Train()
        rewards = ppo.Evaluate()
        Plot(rewards, 'reward', 0, False)
