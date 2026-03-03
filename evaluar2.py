import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from mi_entorno_3sat_observacion import Entorno3SAT

# ==========================================
# 1. CONFIGURACIÓN DE CASOS DE LABORATORIO (40 Agentes)
# ==========================================
def generar_casos_laboratorio(num_variables, num_agentes=40):
    casos = {}
    
    # Utopía: Todos quieren que la ley 0 se apruebe (1), la 1 se rechace (0), etc.
    utopia = {f"agente_{i}": [(0, 1), (1, 0), (2, 0)] for i in range(num_agentes)}
    casos["Caso A - Utopia (Facil)"] = utopia

    # Conflicto: La mitad quiere aprobar la ley 0, la otra mitad quiere rechazarla
    conflicto = {}
    for i in range(num_agentes):
        if i < num_agentes // 2:
            conflicto[f"agente_{i}"] = [(0, 1), (1, 0), (2, 0)]
        else:
            conflicto[f"agente_{i}"] = [(0, 0), (1, 0), (2, 0)]
    casos["Caso B - Conflicto (Medio)"] = conflicto

    return casos

# ==========================================
# 2. FUNCIONES DE ANÁLISIS Y GRÁFICAS
# ==========================================
def graficar_evolucion(matriz_votos, titulo, exito):
    try:
        # Aumentamos el tamaño vertical para que quepan los 40 agentes
        plt.figure(figsize=(10, 10)) 
        
        # Mapa de calor sin números dentro (annot=False) para no saturar la vista
        ax = sns.heatmap(matriz_votos, cmap="RdYlGn", center=0, vmin=-1, vmax=1, 
                         annot=False, cbar_kws={'label': 'Voto (-1 Contra, +1 A favor)'})
        
        estado = '🏆 ÉXITO' if exito else '💀 FRACASO'
        plt.title(f"{titulo} | Resultado: {estado}")
        plt.xlabel("Variables (Leyes)")
        plt.ylabel("Agentes (0 al 39)")
        
        plt.tight_layout()
        print(f"   > Abriendo gráfica para: {titulo}...")
        plt.show()
    except Exception as e:
        print(f"⚠️ Error al generar gráfica: {e}")

def ejecutar_partida(env_raw, model, caso_datos=None):
    options = {"problema_inyectado": caso_datos} if caso_datos else None
    obs_dict, _ = env_raw.reset(options=options)
    
    acciones_dict = {}
    votos_reales = []

    # Solo hay un turno, leemos las acciones de todos
    for agent in env_raw.agents:
        obs_agente = obs_dict[agent]
        action, _ = model.predict(obs_agente, deterministic=True)
        acciones_dict[agent] = action
        
        # Eliminada la conversión > 0.5. MultiDiscrete ya da 0 o 1.
        voto_real = (action * 2) - 1 
        votos_reales.append(voto_real)

    # Avanzamos y terminamos el entorno puro
    obs_dict, rewards, terms, truncs, infos = env_raw.step(acciones_dict)
    
    es_exito = all(r >= 100.0 for r in rewards.values()) if rewards else False
    
    # La matriz ahora guarda las 40 filas (agentes) y 10 columnas (leyes) de un solo turno
    matriz_votos = np.array(votos_reales) 
    
    return matriz_votos, es_exito

# ==========================================
# 3. FUNCIÓN PRINCIPAL
# ==========================================
def evaluar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = "ppo_3sat_final_40agentes.zip"
    ruta_modelo = os.path.join(BASE_DIR, "modelos", nombre_archivo)
    
    print(f"🔍 Buscando cerebro en: {ruta_modelo}")

    if not os.path.exists(ruta_modelo + ".zip") and not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No encuentro el archivo.")
        return

    # Ajustado a 40 agentes
    env_raw = Entorno3SAT(num_agentes=40, num_variables=10)

    model = PPO.load(ruta_modelo)
    print("✅ ¡Modelo cargado! Vamos a examinarlo de verdad.")

    print("\n📊 --- FASE 1: VISUALIZACIÓN ---")
    casos = generar_casos_laboratorio(10, 40)
    
    for nombre_caso, datos_caso in casos.items():
        matriz_votos, exito = ejecutar_partida(env_raw, model, datos_caso)
        graficar_evolucion(matriz_votos, nombre_caso, exito)

    print("\n📈 --- FASE 2: ESTADÍSTICAS GLOBALES (100 PARTIDAS ALEATORIAS) ---")
    total_partidas = 100
    wins = 0
    
    for i in range(total_partidas):
        _, exito = ejecutar_partida(env_raw, model)
        if exito: wins += 1
        if i % 10 == 0: print(".", end="", flush=True)

    print(f"\n\nRESULTADOS DEL MODELO ACTUAL:")
    print(f"---------------------------------------")
    print(f"✅ Tasa de Éxito:       {wins}/{total_partidas} ({wins}%)")
    print(f"---------------------------------------")

if __name__ == "__main__":
    evaluar()