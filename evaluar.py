import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from mi_entonrno_3sat_recompensayobservaciones import Entorno3SAT

# ==========================================
# 1. CONFIGURACIÓN DE CASOS DE LABORATORIO
# ==========================================
def generar_casos_laboratorio(num_variables):
    casos = {}
    casos["Caso A - Utopia (Facil)"] = {
        "agente_0": [(0, 1), (1, 0), (2, 0)], 
        "agente_1": [(0, 1), (3, 0), (4, 0)],
        "agente_2": [(0, 1), (5, 0), (6, 0)],
        "agente_3": [(0, 1), (7, 0), (8, 0)],
        "agente_4": [(0, 1), (9, 0), (1, 1)]
    }
    casos["Caso B - Conflicto (Medio)"] = {
        "agente_0": [(0, 1), (1, 0), (2, 0)], 
        "agente_1": [(0, 1), (3, 0), (4, 0)], 
        "agente_2": [(0, 0), (5, 0), (6, 0)], 
        "agente_3": [(0, 0), (7, 0), (8, 0)], 
        "agente_4": [(0, 1), (9, 0), (1, 1)] 
    }
    return casos

# ==========================================
# 2. FUNCIONES DE ANÁLISIS Y GRÁFICAS
# ==========================================
def graficar_evolucion(historial_votos, titulo, exito):
    try:
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(historial_votos, cmap="RdYlGn", center=0, vmin=-5, vmax=5, 
                         annot=True, fmt=".0f", cbar_kws={'label': 'Suma de Votos'})
        
        estado = '🏆 ÉXITO' if exito else '💀 FRACASO'
        plt.title(f"{titulo} | Resultado: {estado}")
        plt.xlabel("Variables (Leyes)")
        plt.ylabel("Tiempo (Sondeos)")
        
        pasos = range(1, len(historial_votos) + 1)
        ax.set_yticklabels([f"Paso {p}" for p in pasos], rotation=0)
        
        plt.tight_layout()
        print(f"   > Abriendo gráfica para: {titulo}...")
        plt.show()
    except Exception as e:
        print(f"⚠️ Error al generar gráfica: {e}")

# --- CAMBIO CLAVE: BYPASS A LA LIBRERÍA ---
def ejecutar_partida(env_raw, model, caso_datos=None):
    options = {"problema_inyectado": caso_datos} if caso_datos else None
    
    # El entorno puro sí acepta las opciones
    obs_dict, _ = env_raw.reset(options=options)
    
    historial_votos = [] 
    
    # Mientras queden agentes vivos en la partida (Paso 1 al 5)
    while env_raw.agents:
        acciones_dict = {}
        votos_reales_paso = []

        for agent in env_raw.agents:
            obs_agente = obs_dict[agent]
            # PPO predice. Le pasamos la observación de este agente.
            action, _ = model.predict(obs_agente, deterministic=True)
            acciones_dict[agent] = action
            
            # Calculamos el voto físico (-1 o +1) para la gráfica
            accion_binaria = (action > 0.5).astype(int)
            voto_real = (accion_binaria * 2) - 1
            votos_reales_paso.append(voto_real)

        # Sumamos el tablero
        suma_votos_paso = np.sum(votos_reales_paso, axis=0)
        historial_votos.append(suma_votos_paso)

        # Avanzamos un turno en el entorno puro
        obs_dict, rewards, terms, truncs, infos = env_raw.step(acciones_dict)
    
    # En el entorno v2, el máximo son 100 puntos (20pts x 5 cláusulas)
    es_exito = all(r >= 100.0 for r in rewards.values()) if rewards else False
    matriz_historial = np.array(historial_votos)
    
    primer_voto = matriz_historial[0] if len(matriz_historial) > 0 else np.zeros(10)
    ultimo_voto = matriz_historial[-1] if len(matriz_historial) > 0 else np.zeros(10)
    hubo_cambios = not np.array_equal(primer_voto, ultimo_voto)
    
    return matriz_historial, es_exito, hubo_cambios

# ==========================================
# 3. FUNCIÓN PRINCIPAL
# ==========================================
def evaluar():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = "ppo_3sat_final_v1.zip"
    ruta_modelo = os.path.join(BASE_DIR, "modelos", nombre_archivo)
    
    print(f"🔍 Buscando cerebro en: {ruta_modelo}")

    if not os.path.exists(ruta_modelo + ".zip") and not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No encuentro el archivo.")
        return

    # --- CAMBIO CLAVE: Entorno CRUDO sin SuperSuit ---
    env_raw = Entorno3SAT(num_agentes=5, num_variables=10)

    model = PPO.load(ruta_modelo)
    print("✅ ¡Modelo cargado! Vamos a examinarlo de verdad.")

    print("\n📊 --- FASE 1: VISUALIZACIÓN ---")
    casos = generar_casos_laboratorio(10)
    
    for nombre_caso, datos_caso in casos.items():
        historial, exito, cambio = ejecutar_partida(env_raw, model, datos_caso)
        graficar_evolucion(historial, nombre_caso, exito)

    print("\n📈 --- FASE 2: ESTADÍSTICAS GLOBALES (100 PARTIDAS ALEATORIAS) ---")
    total_partidas = 100
    wins = 0
    negociaciones = 0
    
    for i in range(total_partidas):
        _, exito, cambio = ejecutar_partida(env_raw, model)
        if exito: wins += 1
        if cambio: negociaciones += 1
        if i % 10 == 0: print(".", end="", flush=True)

    print(f"\n\nRESULTADOS DEL MODELO ACTUAL:")
    print(f"---------------------------------------")
    print(f"✅ Tasa de Éxito:       {wins}/{total_partidas} ({wins}%)")
    print(f"🤝 Tasa de Negociación: {negociaciones}/{total_partidas} ({negociaciones}%)")
    print(f"---------------------------------------")

if __name__ == "__main__":
    evaluar()