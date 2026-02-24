import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiBinary
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v2"} # Actualizado a v2

    def __init__(self, num_agentes=5, num_variables=10):
        # 1. Parche para SuperSuit
        self.render_mode = None
        
        # --- CONFIGURACIÓN DEL TABLERO ---
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        # --- ACCIONES ---
        self.action_spaces = {
            agent: Box(low=0, high=1, shape=(self.num_variables,), dtype=np.float32)
            for agent in self.possible_agents 
        }

        # --- OBSERVACIÓN (ACTUALIZADA V2) ---
        # CAMBIO: Cada agente solo ve sus propias 3 leyes (3 leyes * 2 datos = 6)
        tamano_clausulas_propio = 6
        
        # Vector DNI (5) + Tablero (10) + Mis Leyes (6) = 21
        tamano_obs = self.num_agentes + self.num_variables + tamano_clausulas_propio

        self.observation_spaces = {
            agent: Box(low=-float("inf"), high=float("inf"), shape=(tamano_obs,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    # --- RESET ---
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_pasos = 0
        self.estado_votacion = np.zeros(self.num_variables, dtype=np.float32)
        self.clausulas_privadas = {}

        if options is not None and "problema_inyectado" in options:
            self.clausulas_privadas = options["problema_inyectado"]
        else:
            for agent in self.agents:
                vars_interes = random.sample(range(self.num_variables), 3)
                signos = [random.choice([0, 1]) for _ in range(3)]
                self.clausulas_privadas[agent] = list(zip(vars_interes, signos))

        observations = {}
        for agent in self.agents:
            observations[agent] = self._crear_observacion(agent)

        return observations, {}

    def step(self, actions):
        self.num_pasos += 1
        
        votos_ronda = np.zeros(self.num_variables, dtype=np.float32)
        
        for agent_id, accion_agente in actions.items():
            accion_binaria = (accion_agente > 0.5).astype(int)
            voto_matematico = (accion_binaria * 2) - 1 
            votos_ronda += voto_matematico

        self.estado_votacion = votos_ronda
        
        LIMITE_PASOS = 5 
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False 
        
        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            resultado_final_leyes = (self.estado_votacion > 0).astype(int)
            
            # CAMBIO V2: Bono Cooperativo Global (MAX-3SAT)
            clausulas_satisfechas_totales = 0
            
            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                for variable_idx, deseo_agente in clausula:
                    if resultado_final_leyes[variable_idx] == deseo_agente:
                        clausulas_satisfechas_totales += 1
                        break # Lógica OR repetada
            
            # Recompensa Global (20 pts por cada cláusula satisfecha en total)
            recompensa_cooperativa = clausulas_satisfechas_totales * 20.0
            
            for agent in self.agents:
                rewards[agent] = recompensa_cooperativa

        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
            
        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # CAMBIO V2: Observación recortada y sin ruido
    def _crear_observacion(self, agent):
        # 1. DNI del Agente (One-hot encoding)
        agente_idx = int(agent.split("_")[1])
        id_one_hot = [0.0] * self.num_agentes
        id_one_hot[agente_idx] = 1.0

        # 2. Datos Públicos (Tablero)
        datos_publicos = self.estado_votacion.tolist()

        # 3. CAMBIO: Datos Privados (SOLO mis 3 leyes)
        datos_mis_clausulas = []
        clausula = self.clausulas_privadas[agent]
        for var_idx, signo in clausula:
            datos_mis_clausulas.extend([var_idx, signo])
        
        # Unimos: DNI (5) + Tablero (10) + Mis Leyes (6) = Vector limpio de 21
        return np.array(id_one_hot + datos_publicos + datos_mis_clausulas, dtype=np.float32)