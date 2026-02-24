import functools
import random
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiBinary
from pettingzoo import ParallelEnv

class Entorno3SAT(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "voto_3sat_v1"}

    def __init__(self, num_agentes=5, num_variables=10):
        # 1. Parche para SuperSuit
        self.render_mode = None
        
        # --- CONFIGURACIÓN DEL TABLERO ---
        self.num_agentes = num_agentes
        self.num_variables = num_variables
        self.possible_agents = [f"agente_{i}" for i in range(num_agentes)]

        # --- ACCIONES ---
        self.action_spaces = {#Es un diccionario con clave los agentes yvalor una box
            agent: Box(low=0, high=1, shape=(self.num_variables,), dtype=np.float32)#Tiene que escupir un float por cada variable (luego en step transformo este float a int en funcion de si es mayor o menor de 0.5)
            for agent in self.possible_agents 
        }

        # --- OBSERVACIÓN ---
        tamano_clausulas_total = self.num_agentes * 6
        
        # CAMBIO CLAVE: Sumamos num_agentes para el vector DNI (One-Hot Encoding)
        tamano_obs = self.num_agentes + self.num_variables + tamano_clausulas_total

        self.observation_spaces = {#Otro diccionario con clave los agentes y valor otra box
            agent: Box(low=-float("inf"), high=float("inf"), shape=(tamano_obs,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)#lo mete en la cache para que sea mas eficiente porqeu lo va a estar preguntando todo el rato
    def observation_space(self, agent):#Esto son gets obligados por pettingzoo
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

        if options is not None and "problema_inyectado" in options:#modo evaluar
            self.clausulas_privadas = options["problema_inyectado"]#acceso al valor del diccionario cuya clave es "problema_inyectado"
        else:#modo entrenamiento
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
            accion_binaria = (accion_agente > 0.5).astype(int)#transforma el voto en un numero entero basicamente
            voto_matematico = (accion_binaria * 2) - 1 #lo transforma en -1 o 1 para las graficas
            votos_ronda += voto_matematico

        self.estado_votacion = votos_ronda
        
        LIMITE_PASOS = 5 
        terminado = (self.num_pasos >= LIMITE_PASOS)
        truncado = False 
        
        rewards = {agent: 0.0 for agent in self.agents}

        if terminado:
            resultado_final_leyes = (self.estado_votacion > 0).astype(int) #si es mayor que 0 devuelve uno (aprobada) eoc 0 (rechazada)
            
            for agent in self.agents:
                clausula = self.clausulas_privadas[agent]
                satisfecho = False
                
                for variable_idx, deseo_agente in clausula:
                    if resultado_final_leyes[variable_idx] == deseo_agente:
                        satisfecho = True
                        break 
                
                if satisfecho:
                    rewards[agent] = 100.0
                else:
                    rewards[agent] = 0.0
        #Puras formalidades para pettinzoo:
        terminations = {agent: terminado for agent in self.agents}
        truncations = {agent: truncado for agent in self.agents}
        infos = {agent: {} for agent in self.agents}#una cosa que no uso asi que pongo diccionarios vacios
        
        observations = {agent: self._crear_observacion(agent) for agent in self.agents}
            
        if terminado:
            self.agents = []

        return observations, rewards, terminations, truncations, infos #estanar de pettingzoo

    # CAMBIO CLAVE: Nueva función para crear la observación con el DNI
    def _crear_observacion(self, agent):
        # 1. DNI del Agente (One-hot encoding: [1,0,0,0,0] = Agente 0)
        agente_idx = int(agent.split("_")[1])
        id_one_hot = [0.0] * self.num_agentes
        id_one_hot[agente_idx] = 1.0

        # 2. Datos Públicos
        datos_publicos = self.estado_votacion.tolist()

        # 3. Datos Globales (Cláusulas de todos)
        datos_clausulas_globales = []
        for nombre_agente in self.possible_agents:
            clausula = self.clausulas_privadas[nombre_agente]
            for var_idx, signo in clausula:
                datos_clausulas_globales.extend([var_idx, signo])
        
        # Unimos: DNI + Tablero + Cláusulas
        return np.array(id_one_hot + datos_publicos + datos_clausulas_globales, dtype=np.float32)