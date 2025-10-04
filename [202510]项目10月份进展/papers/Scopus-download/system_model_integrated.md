## III. 系统模型

### A. MEC环境与深度网络任务图

我们考虑一个**多用户MEC环境**，多个终端设备（Mobile Devices, MDs）将部分DNN推理任务卸载到边缘服务器（ESs）。每个DNN任务被建模为**有向无环图（DAG）**，以捕捉层间依赖和多模态融合阶段。

#### 1. 交通DAG与特征定义

为了让模型具有现实性，我们绑定到一个**具体城市场景**（如北京市中心）。DAG节点代表**交通DNN子任务**，例如：

- 数据采集（摄像头、路侧雷达）
- 车辆轨迹预测
- 信号灯优化计算
- 路径规划与下发

每个节点的计算量、带宽需求和延迟要求根据实际测量或合理假设量化。

| 特征             | 符号      | 描述                          | 单位/类型      |
| ---------------- | --------- | ---------------------------- | ------------- |
| 计算需求         | $c_i$     | 节点i所需CPU周期              | cycles        |
| 内存占用         | $m_i$     | 计算过程中内存使用            | MB            |
| 任务紧急度       | $p_i$     | QoS敏感性或截止时间权重       | scalar        |
| 输出大小         | $d_i^out$ | 传输特征大小                  | KB            |
| 模态             | $μ_i$     | 输入模态（视频、激光雷达）     | categorical   |
| 融合类型         | $φ_i$     | 早期/晚期/混合融合            | categorical   |
| 截止时间/延迟    | $τ_i$     | 预期任务完成时间              | s             |
| 交通负载         | $ρ_i^t$   | 动态交通负载                  | scalar        |

边E表示**任务依赖**，权重由交通仿真或历史数据得到。

| 属性             | 符号        | 描述                           |
| ---------------- | ----------- | ----------------------------- |
| 前驱节点          | u           | 源节点                         |
| 后继节点          | v           | 目标节点                       |
| 传输大小          | $d_uv^trans$ | 需要传输的特征量              |
| 延迟             | $l_uv$      | 预估传输时间                   |
| 可靠性           | $r_uv$      | 数据传输可靠性                 |
| 依赖类型          | $δ_uv$      | 强/弱依赖                      |

形式化表示：

$$
G^t = (V^t, E^t), \quad V^t = \{v_1, v_2, ..., v_N\}, \quad E^t \subseteq V^t \times V^t
$$

DAG快照随时间动态更新，以反映交通流变化。

---

### B. 决策变量

引入六个核心决策变量：

1. **卸载比例** \($\lambda_i \in [0,1]$\)
2. **模型切分层** \($L_i \in \{1,...,N-1\}$\)
3. **设备-边缘关联** \($\xi_{i,s} \in \{0,1\}$\)
4. **CPU资源分配** \($f_{i,s} \in \mathbb{R}^+$\)
5. **语义压缩比** \($\eta_i \in [\eta_{min},\eta_{max}]$\)
6. **内存更新策略** \($u_i \in \{0,1,2\}$\)

---

### C. 目标函数

**多目标奖励函数**：

$$
R = \alpha R_T + \beta R_E + \gamma R_B + \delta R_P
$$

- 延迟降低 \($R_T$\)
- 能耗节省 \($R_E$\)
- 带宽惩罚 \($R_B$\)
- 隐私风险 \($R_P$\)

权重 \($\alpha,\beta,\gamma,\delta$\) 控制每个目标的优先级。

---

### D. 约束条件

1. 边缘服务器计算能力：\($\sum_i f_{i,s} \le F_s^{max}$\)
2. 上行带宽约束：\($\sum_{i:\xi_{i,s}=1} \eta_i \cdot D_i \le B_s^{max}$\)
3. DAG依赖约束：\($\text{Start}(v) \ge \max_{u \in \text{prec}(v)} \text{Finish}(u)$\)
4. 模型切分可行性：\($L_i \in \mathcal{L}_{valid}$\)
5. 内存容量：\($|Q_i| \le Q_{max}$\)

---

### E. 强化学习建模

#### 1. MADDPG 状态-动作-奖励定义

- **状态空间 s**：通过DGNN得到的节点嵌入 + 任务进度 + 内存队列状态 + 服务器负载 + 邻居节点信息
- **动作空间 a**：每个智能体（MEC节点/交通路口）可选择
  - 本地执行任务
  - 卸载到邻居MEC节点
  - 卸载到云端
- **奖励函数 R**：多目标加权组合

$$
R = \alpha (\text{latency}) + \beta (\text{energy}) + \gamma (\text{success rate})
$$

智能体通过CTDE训练：集中式Critic看到全局状态/动作，Actor根据本地观测去分布式执行。

#### 2. DGNN输入特征与时序更新

**节点特征** \($x_i^t$\)：计算需求、内存占用、任务紧急度、输出大小、模态、融合类型、截止时间、动态交通负载

**边特征** \($e_{ij}^t$\)：传输大小、延迟、可靠性、依赖类型

**时序更新规则**：每个时间槽的DAG快照 \($G^t$\)，节点与边特征随时间更新：
$$
x_i^{t+1} = f_{node}(x_i^t, traffic_i^t, task\_arrival_i^t)$$
$$
$$
e_{ij}^{t+1} = f_{edge}(e_{ij}^t, network\_load_{ij}^t)
$$

节点嵌入：\($h_i^t = DGNN(x_i^t, e_{ij}^t, G^t)$\)

时间编码：\($\tilde{h}_i^t = GRU(h_i^t, \tilde{h}_i^{t-1})$\)

#### 3. DGNN接口规范

```python
class DAGEncoder(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # 节点嵌入供RL智能体使用
```

#### 4. MADDPG智能体接口

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 动作映射到 [-1,1]
```

**环境模拟器**：
- DAG任务管理（任务进度、依赖关系）
- 计算/传输资源（CPU、带宽、延迟）
- 语义内存管理（队列、压缩）
- 动态网络/交通更新

---

### F. 马尔科夫决策过程（MDP）形式化

将MEC + DAG任务卸载系统形式化为MDP（或POMDP）：

- **状态 $s^t$**：\($s^t = \{x_i^t, e_{ij}^t, task\_queue_i^t, server\_load_s^t\}$\)
- **动作 $a^t$**：每个智能体的选择（本地执行、卸载到邻居MEC、卸载到云端）
- **转移概率 $P(s^{t+1}|s^t, a^t)$**：由以下因素决定
  - 任务执行成功/失败概率（依赖服务器容量、队列长度、网络带宽）
  - 通信延迟及潜在丢包
  - DAG依赖满足情况（前驱任务完成后才可执行）
- **奖励 R^t**：延迟、能耗、成功率、隐私风险的加权和
- **观测 o^t**：每个智能体可用的局部观测（仅本地信息时为POMDP；全局可见时为MDP）

**状态更新公式**：
$$
x_i^{t+1} = f_{node}(x_i^t, a_i^t, traffic_i^t, task\_arrival_i^t)
$$
$$
e_{ij}^{t+1} = f_{edge}(e_{ij}^t, a_i^t, network\_load_{ij}^t)
$$

$$
task\_queue_i^{t+1} = task\_queue_i^t - completed_i^t + new\_arrival_i^t
$$
$$
server\_load_s^{t+1} = server\_load_s^t + assigned\_tasks_s^t - completed_s^t
$$

该形式化为DGNN + MADDPG RL设置与基于MDP的优化框架提供了桥梁。

