import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiDiscrete
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v4_egoista"} 

    def __init__(self, num_agentes=40, num_variables=10):
        self.render_mode = None
        
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        self.action_spaces = {
            agent: MultiDiscrete([2] * self.num_variables)
            for agent in self.possible_agents 
        }
        # --- OBSERVACIÓN (V3: Mapa Posicional) ---
        tamano_clausulas_propio = self.num_variables
        tamano_obs = self.num_agentes  + tamano_clausulas_propio #ve su numero de agente y su clausula

        self.observation_spaces = {#Crea la observacion para cada agente
            agent: Box(low=-float("inf"), high=float("inf"), shape=(tamano_obs,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
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
            voto_matematico = (accion_agente * 2) - 1 
            votos_ronda += voto_matematico

        self.estado_votacion = votos_ronda
        
        LIMITE_PASOS = 1 
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False
        
        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            resultado_final_leyes = (self.estado_votacion > 0).astype(int)
            
            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                satisfecho = False
                
                for variable_idx, deseo_agente in clausula:
                    if resultado_final_leyes[variable_idx] == deseo_agente:
                        satisfecho = True
                        break
                
                rewards[agent] = 100.0 if satisfecho else 0.0
                
        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
            
        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
    # --- OBSERVACIÓN (V3: Mapa Posicional) ---
    def _crear_observacion(self, agent):
        agente_idx = int(agent.split("_")[1])
        id_one_hot = [0.0] * self.num_agentes
        id_one_hot[agente_idx] = 1.0 #la lista con el DNI


        mapa_leyes = [0.0] * self.num_variables
        clausula = self.clausulas_privadas[agent]
        for var_idx, signo in clausula:
            mapa_leyes[var_idx] = 1.0 if signo == 1 else -1.0
        
        return np.array(id_one_hot  + mapa_leyes, dtype=np.float32)