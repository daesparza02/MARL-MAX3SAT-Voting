# MARL MAX-3SAT Voting Simulator

Este proyecto implementa un entorno de Reinforcement Learning Multi-Agente (MARL) donde 5 inteligencias artificiales negocian y votan sobre 10 variables (leyes) basándose en el problema matemático MAX-3SAT. 

Desarrollado como parte de un Trabajo de Fin de Grado (TFG), el modelo demuestra cómo agentes independientes pueden aprender a cooperar, ceder en variables que no les afectan y maximizar el beneficio global a través de pactos implícitos.

## Características Principales

* **Entorno Customizado:** Creado desde cero usando `PettingZoo` (ParallelEnv) y `Gymnasium`.
* **Algoritmo:** Entrenamiento basado en Proximal Policy Optimization (PPO) usando `Stable-Baselines3`.
* **Representación Posicional (v3):** Las observaciones de los agentes utilizan un "mapa posicional" de 10 espacios (1, 0, -1) que resuelve la ceguera de las redes MLP estándar ante etiquetas numéricas.
* **Recompensa Cooperativa:** Transición de un modelo egoísta a uno de beneficio global (MAX-3SAT), forzando a la red a descubrir la negociación.
* **Visualización:** Generación automática de mapas de calor (Heatmaps) con `Seaborn` para auditar las votaciones paso a paso.

##  Requisitos e Instalación

Asegúrate de tener Python 3.8+ instalado. Las librerías principales son:

```bash
pip install gymnasium pettingzoo supersuit stable-baselines3 numpy matplotlib seaborn
