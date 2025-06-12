# 🚗 Autonomous Vehicle Safety & Navigation through AI

<div align="center">
  <img src="https://i.ytimg.com/vi/u2TxYhv3UKE/maxresdefault.jpg" alt="CARLA Simulation Environment" width="100%" style="border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin: 20px 0;">
  
  <h3 style="color: #4A90E2; margin: 0; font-weight: 300;">Advanced Autonomous Vehicle Control Systems in CARLA Simulation</h3>
  <p style="color: #666; font-size: 18px; margin: 10px 0;">Demonstrating Progressive AI Architectures for Autonomous Navigation</p>
</div>

<div align="center">
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
  ![CARLA Version](https://img.shields.io/badge/CARLA-0.10.0-00D2FF?style=for-the-badge&logo=unrealengine&logoColor=white)
  ![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![Development](https://img.shields.io/badge/Development-Active-00C851?style=for-the-badge&logo=checkmarx&logoColor=white)

</div>

---

## 🎯 Project Overview

This project implements **three progressive autonomous vehicle control architectures** within the CARLA simulator, each representing an evolutionary step in autonomous driving technology. From foundational P-control systems to sophisticated AI-driven decision making, this implementation showcases the technological progression in autonomous vehicle safety and navigation.

<div align="center">
  <img src="https://img.shields.io/badge/CARLA-Open%20Source%20Simulator-00D2FF?style=flat&logo=unrealengine&logoColor=white" alt="CARLA Badge" />
  <br>
  <em>An open-source autonomous driving simulator built on Unreal Engine 4, providing photorealistic urban environments with dynamic weather, lighting conditions, and complex traffic scenarios for robust autonomous vehicle testing.</em>
</div>

<br>

<div align="center">
  <table style="border: none; width: 100%; max-width: 1000px;">
    <tr>
      <td align="center" style="border: none; padding: 30px; width: 33.33%;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center;">
          <h3 style="margin: 0 0 15px 0; font-size: 1.3em;">🎯 Basic Navigation</h3>
          <h4 style="margin: 0 0 10px 0; font-weight: 600;">A* Pathfinding + P Control</h4>
          <p style="margin: 0; opacity: 0.9; line-height: 1.4;">Foundational autonomous navigation with proportional control for steering and throttle</p>
        </div>
      </td>
      <td align="center" style="border: none; padding: 30px; width: 33.33%;">
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; padding: 25px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center;">
          <h3 style="margin: 0 0 15px 0; font-size: 1.3em;">🎯 MPC Enhanced</h3>
          <h4 style="margin: 0 0 10px 0; font-weight: 600;">Model Predictive Control</h4>
          <p style="margin: 0; opacity: 0.9; line-height: 1.4;">Advanced trajectory optimization with predictive horizon planning</p>
        </div>
      </td>
      <td align="center" style="border: none; padding: 30px; width: 33.33%;">
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; padding: 25px; color: white; height: 200px; display: flex; flex-direction: column; justify-content: center;">
          <h3 style="margin: 0 0 15px 0; font-size: 1.3em;">🎯 AI-Powered Navigation</h3>
          <h4 style="margin: 0 0 10px 0; font-weight: 600;">MPC + Reinforcement Learning</h4>
          <p style="margin: 0; opacity: 0.9; line-height: 1.4;">Intelligent decision making with PPO-based adaptive control</p>
        </div>
      </td>
    </tr>
  </table>
</div>

---

## 🔬 Technical Implementations

### 📊 **System Architecture**

<div align="center">

```mermaid
graph TB
    subgraph "🌍 CARLA Simulation Environment"
        A[🏙️ Urban Environment] --> B[📊 Sensor Suite]
        B --> C[🚗 Ego Vehicle]
        C --> D[🚦 Dynamic Scenarios]
    end
    
    subgraph "🎛️ Control Systems"
        E{🧠 Controller Selection}
        E -->|Basic| F[🗺️ A* Pathfinding]
        E -->|Enhanced| G[📈 MPC Optimization]
        E -->|AI-Powered| H[🤖 RL Decision Engine]
    end
    
    subgraph "📡 Perception Layer"
        I[📷 RGB Camera]
        J[🌊 LiDAR]
        K[📡 Radar]
        L[👁️ YOLO Detection]
    end
    
    subgraph "⚙️ Control Algorithms"
        F --> M[🎯 P Controller]
        G --> N[📊 MPC Controller]
        H --> O[🧠 PPO Agent]
    end
    
    subgraph "🛡️ Safety Systems"
        P[🚨 Collision Avoidance]
        Q[🚶 Pedestrian Detection]
    end
    
    B --> I & J & K
    I & J & K --> L
    L --> E
    M & N & O --> P & Q
    P & Q --> R[🚗 Vehicle Control Commands]
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style L fill:#fff3e0
    style R fill:#e8f5e8
```

</div>

---

### 🚙 **1. Basic Navigation Controller**

**Foundation System** - Implements core autonomous driving principles using classical control theory and pathfinding algorithms.

#### **Mathematical Framework**

<div align="center">

**P-Controller for Steering:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\delta(t)=K_p\times{e(t)}" title="P-Controller" />

**Where:**
- δ(t) = steering angle
- K<sub>p</sub> = proportional gain constant  
- e(t) = lateral error from desired path

**A* Pathfinding Cost Function:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f(n)=g(n)+h(n)" title="A-Star Cost" />

**Where:**
- g(n) = actual cost from start to node n
- h(n) = heuristic estimate from n to goal

</div>

<details>
<summary><strong>🔧 Technical Specifications</strong></summary>

```python
CONTROLLER_CONFIG = {
    'target_speed': 30,          # km/h
    'safety_distance': 10,       # meters
    'control_frequency': 10,     # Hz
    'detection_range': 100,      # meters
    'steering_gain': 0.8,        # Kp for steering
    'throttle_gain': 0.5,        # Kp for speed control
    'brake_threshold': 0.3       # Emergency braking threshold
}
```

**Core Capabilities:**
- ✅ Waypoint-based navigation using A* pathfinding
- ✅ Proportional control for steering and throttle
- ✅ Basic obstacle detection and avoidance
- ✅ NPC vehicle and pedestrian interaction

</details>

---

### 🚘 **2. MPC Navigation Controller**

**Enhanced Control System** - Leverages Model Predictive Control for optimal trajectory planning with predictive horizon optimization.

#### **MPC Mathematical Formulation**

<div align="center">

**Discrete-Time Vehicle Model:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{x}(k+1)=f(\mathbf{x}(k),\mathbf{u}(k))" title="Vehicle Model" />

**Optimization Objective:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\min\sum_{k=0}^{N-1}\left[\|\mathbf{x}(k)-\mathbf{x}_{ref}(k)\|_{\mathbf{Q}}^2+\|\mathbf{u}(k)\|_{\mathbf{R}}^2\right]" title="MPC Objective" />

**State Vector:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{x}=[p_x,p_y,\psi,v]^T" title="State Vector" />

**Where:**
- p<sub>x</sub>, p<sub>y</sub> = position coordinates
- ψ = heading angle  
- v = velocity

**Control Vector:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{u}=[\delta,a]^T" title="Control Vector" />

**Where:**
- δ = steering angle
- a = acceleration

</div>

<details>
<summary><strong>⚙️ MPC Configuration</strong></summary>

```python
MPC_PARAMETERS = {
    'prediction_horizon': 10,    # N steps ahead
    'control_horizon': 5,        # Control moves
    'sampling_time': 0.1,        # seconds
    'optimization_solver': 'ipopt',
    'max_steering_angle': 0.5,   # radians
    'max_acceleration': 3.0,     # m/s²
    'weight_matrices': {
        'Q': [10, 10, 1, 1],     # State weights
        'R': [1, 1]              # Control weights
    }
}
```

**Performance Improvements:**
- 🔺 50% reduction in steering oscillation
- 🔺 30% smoother acceleration profiles
- 🔺 Enhanced cornering stability
- 🔺 Improved trajectory tracking accuracy

</details>

---

### 🤖 **3. MPC-RL Controller**

**AI-Powered System** - Integrates Reinforcement Learning with MPC for adaptive, intelligent autonomous navigation.

#### **PPO Algorithm Implementation**

<div align="center">

**Policy Optimization Objective:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L^{CLIP}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]" title="PPO Objective" />

**Value Function Loss:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L^{VF}(\theta)=\mathbb{E}\left[\left(V_\theta(s_t)-V_t^{target}\right)^2\right]" title="Value Loss" />

**Combined PPO Objective:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L(\theta)=\mathbb{E}\left[L^{CLIP}(\theta)-c_1L^{VF}(\theta)+c_2H(\pi_\theta)\right]" title="Combined PPO" />

</div>

<details>
<summary><strong>🧠 AI Architecture</strong></summary>

```python
PPO_CONFIG = {
    'algorithm': 'PPO',
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'gamma': 0.99,
    'clip_range': 0.2,
    'entropy_coefficient': 0.01,
    'value_function_coefficient': 0.5,
    'network_architecture': [64, 64],
    'activation_function': 'tanh'
}
```

**Advanced Integration:**
- 🧠 **PPO Agent**: Proximal Policy Optimization for stable learning
- 👁️ **YOLO Detection**: Real-time object detection at 10 FPS
- 🎯 **MPC Integration**: Optimal control with learned policies
- 🌐 **Environmental Adaptation**: Dynamic response to traffic conditions

</details>

---

## 📊 Performance Analysis

<div align="center">

### 🏆 **Comparative Performance Metrics**

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 30px; margin: 20px 0;">

<table style="width: 100%; color: white; border-collapse: collapse;">
<thead>
<tr style="background: rgba(255,255,255,0.1); border-radius: 10px;">
<th style="padding: 15px; text-align: left; font-weight: 600;">Performance Metric</th>
<th style="padding: 15px; text-align: center; font-weight: 600;">Basic Navigation</th>
<th style="padding: 15px; text-align: center; font-weight: 600;">MPC Enhanced</th>
<th style="padding: 15px; text-align: center; font-weight: 600;">AI-Powered</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
<td style="padding: 12px; font-weight: 500;"><strong>Collision Rate</strong></td>
<td style="padding: 12px; text-align: center;">~15%</td>
<td style="padding: 12px; text-align: center;">~10%</td>
<td style="padding: 12px; text-align: center; background: rgba(0,200,81,0.3); border-radius: 5px;"><strong>&lt;5%</strong></td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
<td style="padding: 12px; font-weight: 500;"><strong>Path Smoothness</strong></td>
<td style="padding: 12px; text-align: center;">⭐⭐</td>
<td style="padding: 12px; text-align: center;">⭐⭐⭐</td>
<td style="padding: 12px; text-align: center; background: rgba(0,200,81,0.3); border-radius: 5px;"><strong>⭐⭐⭐⭐⭐</strong></td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
<td style="padding: 12px; font-weight: 500;"><strong>Speed Control Accuracy</strong></td>
<td style="padding: 12px; text-align: center;">±20%</td>
<td style="padding: 12px; text-align: center;">±10%</td>
<td style="padding: 12px; text-align: center; background: rgba(0,200,81,0.3); border-radius: 5px;"><strong>±5%</strong></td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
<td style="padding: 12px; font-weight: 500;"><strong>Traffic Adaptation</strong></td>
<td style="padding: 12px; text-align: center;">Basic</td>
<td style="padding: 12px; text-align: center;">Moderate</td>
<td style="padding: 12px; text-align: center; background: rgba(0,200,81,0.3); border-radius: 5px;"><strong>Advanced</strong></td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
<td style="padding: 12px; font-weight: 500;"><strong>Response Time</strong></td>
<td style="padding: 12px; text-align: center;">100ms</td>
<td style="padding: 12px; text-align: center;">80ms</td>
<td style="padding: 12px; text-align: center; background: rgba(0,200,81,0.3); border-radius: 5px;"><strong>50ms</strong></td>
</tr>
<tr>
<td style="padding: 12px; font-weight: 500;"><strong>Computational Load</strong></td>
<td style="padding: 12px; text-align: center;">Low</td>
<td style="padding: 12px; text-align: center;">Medium</td>
<td style="padding: 12px; text-align: center;">High</td>
</tr>
</tbody>
</table>

</div>

</div>

---

## 🎬 Live Demonstrations

<div align="center">

### **Progressive Control Evolution**

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; max-width: 1000px;">

<div style="background: linear-gradient(145deg, #667eea, #764ba2); border-radius: 15px; padding: 20px; color: white; text-align: center;">
<h4 style="margin: 0 0 15px 0;">🚙 Basic Navigation</h4>
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin: 10px 0;">
<img src="images/stops1.gif" width="100%" style="border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>
<p style="margin: 10px 0 0 0; opacity: 0.9;"><em>P-control with obstacle detection</em></p>
</div>

<div style="background: linear-gradient(145deg, #f093fb, #f5576c); border-radius: 15px; padding: 20px; color: white; text-align: center;">
<h4 style="margin: 0 0 15px 0;">🚘 MPC Enhanced</h4>
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin: 10px 0;">
<img src="images/stops3.gif" width="100%" style="border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>
<p style="margin: 10px 0 0 0; opacity: 0.9;"><em>Predictive trajectory optimization</em></p>
</div>

<div style="background: linear-gradient(145deg, #4facfe, #00f2fe); border-radius: 15px; padding: 20px; color: white; text-align: center;">
<h4 style="margin: 0 0 15px 0;">🤖 AI-Powered</h4>
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin: 10px 0;">
<img src="images/first.gif" width="100%" style="border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>
<p style="margin: 10px 0 0 0; opacity: 0.9;"><em>Adaptive intelligent navigation</em></p>
</div>

</div>

### **Advanced AI Navigation Scenarios**

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0; max-width: 800px;">

<div style="background: linear-gradient(145deg, #667eea, #764ba2); border-radius: 15px; padding: 15px;">
<img src="images/second.gif" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>

<div style="background: linear-gradient(145deg, #4facfe, #00f2fe); border-radius: 15px; padding: 15px;">
<img src="images/third.gif" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>

</div>

</div>

---

## 🚀 Quick Start Guide

### **System Requirements**

<div align="center">

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; margin: 20px 0; max-width: 600px;">

<table style="width: 100%; color: white; border-collapse: collapse;">
<thead>
<tr>
<th style="padding: 12px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Component</th>
<th style="padding: 12px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Minimum</th>
<th style="padding: 12px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Recommended</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 10px; text-align: center; font-weight: 500;"><strong>OS</strong></td>
<td style="padding: 10px; text-align: center;">Windows 10 / Ubuntu 18.04+</td>
<td style="padding: 10px; text-align: center;">Windows 11 / Ubuntu 20.04+</td>
</tr>
<tr>
<td style="padding: 10px; text-align: center; font-weight: 500;"><strong>Python</strong></td>
<td style="padding: 10px; text-align: center;">3.9+</td>
<td style="padding: 10px; text-align: center;">3.12</td>
</tr>
<tr>
<td style="padding: 10px; text-align: center; font-weight: 500;"><strong>RAM</strong></td>
<td style="padding: 10px; text-align: center;">8GB</td>
<td style="padding: 10px; text-align: center;">16GB+</td>
</tr>
<tr>
<td style="padding: 10px; text-align: center; font-weight: 500;"><strong>GPU</strong></td>
<td style="padding: 10px; text-align: center;">GTX 1050</td>
<td style="padding: 10px; text-align: center;">RTX 3060+</td>
</tr>
<tr>
<td style="padding: 10px; text-align: center; font-weight: 500;"><strong>Storage</strong></td>
<td style="padding: 10px; text-align: center;">50GB</td>
<td style="padding: 10px; text-align: center;">100GB SSD</td>
</tr>
</tbody>
</table>

</div>

</div>

### **Installation**

<details>
<summary><strong>🔧 Complete Setup Instructions</strong></summary>

**1. CARLA Simulator Setup**
```bash
# Download CARLA 0.10.0
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.10.0.tar.gz
tar -xzf CARLA_0.10.0.tar.gz

# Launch CARLA server
cd CARLA_0.10.0
./CarlaUE4.sh -quality-level=Low -fps=30
```

**2. Project Installation**
```bash
# Clone repository
git clone https://github.com/Baebabe/Autonomous-Vehicle-Safety-and-Navgiation-through-AI.git
cd Autonomous-Vehicle-Safety-and-Navgiation-through-AI

# Setup virtual environment
python -m venv carla_env
source carla_env/bin/activate  # Linux/Mac
# carla_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**3. CARLA Python API Configuration**
```bash
# Add CARLA Python API to path
export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.10.0/PythonAPI/carla/dist/carla-0.10.0-py3.7-linux-x86_64.egg
```

</details>

### **🎮 Running Simulations**

<div align="center">

<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; padding: 25px; margin: 20px 0; max-width: 700px;">

<table style="width: 100%; color: white; border-collapse: collapse;">
<thead>
<tr>
<th style="padding: 15px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Controller</th>
<th style="padding: 15px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Command</th>
<th style="padding: 15px; text-align: center; font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.3);">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 12px; text-align: center; font-weight: 500;"><strong>🚙 Basic</strong></td>
<td style="padding: 12px; text-align: center; font-family: monospace; background: rgba(255,255,255,0.1); border-radius: 5px;"><code>python controller/main_navigation.py</code></td>
<td style="padding: 12px; text-align: center;">P-control with A* pathfinding</td>
</tr>
<tr>
<td style="padding: 12px; text-align: center; font-weight: 500;"><strong>🚘 MPC</strong></td>
<td style="padding: 12px; text-align: center; font-family: monospace; background: rgba(255,255,255,0.1); border-radius: 5px;"><code>python controller/main_mpc.py</code></td>
<td style="padding: 12px; text-align: center;">Model Predictive Control</td>
</tr>
<tr>
<td style="padding: 12px; text-align: center; font-weight: 500;"><strong>🤖 AI</strong></td>
<td style="padding: 12px; text-align: center; font-family: monospace; background: rgba(255,255,255,0.1); border-radius: 5px;"><code>python controller/main_mpc_rl.py</code></td>
<td style="padding: 12px; text-align: center;">Reinforcement Learning integration</td>
</tr>
</tbody>
</table>

</div>

</div>

---

## 🏗️ Project Structure

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; margin: 20px 0;">

```
📦 Autonomous-Vehicle-AI/
├── 🎮 controller/
│   ├── main_navigation.py          # Basic P-control navigation
│   ├── main_mpc.py                # MPC implementation
│   ├── main_mpc_rl.py             # RL-enhanced MPC
│   ├── vehicle_detector.py        # Vehicle detection system
│   ├── navigation_controller.py   # P-control algorithms
│   ├── mpc_controller.py          # MPC optimization
│   └── safety_controller.py       # Safety protocols
├── 👁️ perception/
│   ├── carla_environment.py       # CARLA interface
│   ├── mpc.py                     # MPC mathematical model
│   └── sb3_ppo_train.py          # PPO training pipeline
├── 📊 models/                     # Trained RL models
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # Documentation
```

</div>

---

## 🧠 AI Training Pipeline

### **Reinforcement Learning Training**

<details>
<summary><strong>🚀 Training Configuration</strong></summary>

**Training Hyperparameters:**
```python
TRAINING_CONFIG = {
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'gamma': 0.99,
    'clip_range': 0.2,
    'entropy_coefficient': 0.01,
    'save_frequency': 50_000,
    'evaluation_frequency': 10_000
}
```

**Start Training:**
```bash
python -c "from controller.main_mpc_rl import train_model; train_model()"
```

**Monitor Training Progress:**
```bash
tensorboard --logdir=./logs/ppo_training
```

</details>

---

## 📚 References & Documentation

### **Academic Foundations**

- **Schulman, J. et al.** *"Proximal Policy Optimization Algorithms"* - arXiv:1707.06347, 2017
- **Rawlings, J.B. et al.** *"Model Predictive Control: Theory and Design"* - Nob Hill Publishing, 2009
- **Hart, P. et al.** *"A Formal Basis for the Heuristic Determination of Minimum Cost Paths"* - IEEE Transactions on Systems, 1968

### **Video Demonstrations**

<div align="center">

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0; max-width: 800px;">

<div style="background: linear-gradient(145deg, #667eea, #764ba2); border-radius: 15px; padding: 20px; text-align: center; color: white; transition: transform 0.3s ease;">
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
<img src="https://img.youtube.com/vi/jG5JvMHZknQ/maxresdefault.jpg" alt="Basic Navigation Demo" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>
<h4 style="margin: 0 0 10px 0;">🚙 Basic Navigation Demo</h4>
<a href="https://youtu.be/jG5JvMHZknQ" style="display: inline-block; background: rgba(255,255,255,0.2); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; font-weight: 600; transition: background 0.3s ease;">
▶️ Watch Demo
</a>
</div>

<div style="background: linear-gradient(145deg, #4facfe, #00f2fe); border-radius: 15px; padding: 20px; text-align: center; color: white; transition: transform 0.3s ease;">
<div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
<img src="https://img.youtube.com/vi/LuupTtgS4D0/maxresdefault.jpg" alt="MPC Controller Demo" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
</div>
<h4 style="margin: 0 0 10px 0;">🚘 MPC Controller Demo</h4>
<a href="https://youtu.be/LuupTtgS4D0" style="display: inline-block; background: rgba(255,255,255,0.2); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; font-weight: 600; transition: background 0.3s ease;">
▶️ Watch Demo
</a>
</div>

</div>

</div>

---

<div align="center">
  
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 40px; margin: 40px 0; max-width: 800px;">
    
    <h3 style="color: white; margin: 0 0 20px 0;">Built with cutting-edge technologies</h3>
    
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px 0;">
      <img src="https://img.shields.io/badge/CARLA-Simulator-00D2FF?style=flat-square&logo=unrealengine" alt="CARLA" />
      <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch" />
      <img src="https://img.shields.io/badge/SB3-Reinforcement%20Learning-4285F4?style=flat-square" alt="Stable Baselines3" />
      <img src="https://img.shields.io/badge/YOLO-Object%20Detection-00FFFF?style=flat-square" alt="YOLO" />
      <img src="https://img.shields.io/badge/CasADi-Optimization-FF6B6B?style=flat-square" alt="CasADi" />
    </div>
    
    <div style="height: 2px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent); margin: 30px 0;"></div>
    
    <p style="color: rgba(255,255,255,0.9); font-style: italic; font-size: 1.1em; margin: 20px 0;">
      Advancing the frontiers of autonomous vehicle technology through intelligent control systems
    </p>
    
    <div style="margin-top: 30px;">
      <a href="https://github.com/Baebabe/Autonomous-Vehicle-Safety-and-Navgiation-through-AI" style="display: inline-block; background: rgba(255,255,255,0.2); color: white; text-decoration: none; padding: 15px 30px; border-radius: 30px; font-weight: 600; font-size: 1.1em; transition: background 0.3s ease;">
        ⭐ Star this repository
      </a>
    </div>
    
  </div>
  
</div>