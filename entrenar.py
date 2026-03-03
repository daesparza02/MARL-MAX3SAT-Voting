import os
import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mi_entorno_3sat_observacion import Entorno3SAT

def entrenar():
    # 1. Configuración de carpetas
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MODEL_DIR = os.path.join(BASE_DIR, "modelos")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. CONFIGURACIÓN DE TIEMPO (FUERZA BRUTA)
    NUM_AGENTES = 40
    NUM_VARIABLES = 10
    
    # 3 MILLONES de pasos. 
    # En un PC normal esto son unas 3-4 horas. 
    # Es suficiente para ver resultados muy sólidos.
    TOTAL_TIMESTEPS = 10_000_000 

    print(f"--- ENTRENAMIENTO BLINDADO (Sin paradas) ---")
    print(f"   > Objetivo: {TOTAL_TIMESTEPS} pasos.")
    print(f"   > Guardando en: {MODEL_DIR}")

    # 3. Entorno
    env = Entorno3SAT(num_agentes=NUM_AGENTES, num_variables=NUM_VARIABLES)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # 4. El Cerebro
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        batch_size=2048,        # Aumentamos el batch para más estabilidad
        n_steps=2048,           # Pasos por actualización
        gamma=0.99,
        ent_coef=0.01           # Vital para que exploren
    )

    # 5. Guardado de seguridad cada 100.000 pasos
    checkpoint_callback = CheckpointCallback(        save_freq=100_000, 
        save_path=MODEL_DIR, 
        name_prefix="ppo_3sat_larga_duracion"
    )

    print("🚀 Entrenando... (Volveré dentro de unas horas)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    # 6. Guardado Final
    nombre_final = "ppo_3sat_final_40agentes" # Usamos el mismo nombre para que evaluar.py lo encuentre fácil
    ruta_final = os.path.join(MODEL_DIR, nombre_final)
    model.save(ruta_final)
    
    print("---------------------------------------------------------")
    print(f"¡TERMINADO! Modelo guardado en: {ruta_final}.zip")
    print("Ahora sí, ejecuta evaluar.py y verás la magia.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    entrenar()
