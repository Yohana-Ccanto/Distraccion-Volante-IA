from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import pandas as pd 

app = Flask(__name__)

# --- 1. Definir las Etiquetas y MAPEO DE RIESGO BASE (MODIFICADO) ---
# Hemos consolidado la información de la distracción, el score base, y el mensaje
LABELS_RISK_DATA = {
    # Formato: { 'score': Puntuación Base (0-100), 'nivel': Nivel de Riesgo, 'impacto': Mensaje de Riesgo, 'tipo': Clasificación CSS }
    0: {'score': 100, 'nivel': 'Bajo', 'impacto': 'Conducción segura. Excelente! La atención plena en la carretera es clave.', 'tipo': 'success', 'name': 'Conducción Normal'},
    1: {'score': 5, 'nivel': 'CRÍTICO', 'impacto': 'Desviar la vista y la mente para enviar mensajes aumenta drásticamente el riesgo. Detente antes de enviar mensajes.', 'tipo': 'error', 'name': 'Enviando mensajes de texto (derecha)'},
    2: {'score': 10, 'nivel': 'ALTO', 'impacto': 'Una llamada sin manos libres duplica la probabilidad de accidente. Cuelga o usa el manos libres.', 'tipo': 'error', 'name': 'Hablando por teléfono (derecha)'},
    3: {'score': 5, 'nivel': 'CRÍTICO', 'impacto': 'La distracción cognitiva, visual y manual que implica enviar mensajes es una de las principales causas de choques fatales. Espera a llegar a tu destino.', 'tipo': 'error', 'name': 'Enviando mensajes de texto (izquierda)'},
    4: {'score': 10, 'nivel': 'ALTO', 'impacto': 'El uso del teléfono sin manos libres reduce el tiempo de reacción al frenado. Cuelga o usa en manos libres.', 'tipo': 'error', 'name': 'Hablando por teléfono (izquierda)'},
    5: {'score': 40, 'nivel': 'Moderado', 'impacto': 'Ajustar controles distrae la vista y las manos. Tómate el tiempo necesario para hacer ajustes solo cuando sea seguro.', 'tipo': 'warning', 'name': 'Operando la radio/climatizador'},
    6: {'score': 30, 'nivel': 'Moderado', 'impacto': 'Beber requiere una mano fuera del volante y distracción visual. Bebe solo en paradas seguras o con extrema precaución.', 'tipo': 'warning', 'name': 'Bebiendo'},
    7: {'score': 15, 'nivel': 'ALTO', 'impacto': 'Desviar la vista de la carretera y girar el torso es extremadamente peligroso. Detente o pídale a un pasajero que lo alcance.', 'tipo': 'error', 'name': 'Alcanzando algo detrás'},
    8: {'score': 20, 'nivel': 'Alto', 'impacto': 'El coche no es un tocador. Esta acción desvía la atención visual y cognitiva. Detente antes de realizar aseo personal.', 'tipo': 'error', 'name': 'Aseo personal (Maquillaje/Pelo)'},
    9: {'score': 60, 'nivel': 'Bajo', 'impacto': 'Conversación intensa puede desviar tu atención cognitiva del tráfico. Evita miradas prolongadas al pasajero.', 'tipo': 'warning', 'name': 'Hablando con el pasajero'}
}

# FACTORES DE RIESGO ADAPTATIVO
RISK_FACTORS = {
    'Principiante': 15,  # Penalización más alta al Score (aumenta el riesgo percibido)
    'Intermedio': 0,     # Sin penalización
    'Avanzado': -5       # Bonificación (disminuye ligeramente el riesgo percibido)
}

def get_prediction_data(probs, confidence_threshold, user_profile):
    """Procesa los resultados de YOLOv8, aplica el umbral y el riesgo adaptativo."""
    
    # 1. Extracción de predicción de YOLO
    top1_index = probs.top1
    top1_confidence = probs.top1conf.item()
    
    # Si la confianza de la clase principal es menor que el umbral del usuario,
    # asumimos 'Conducción Normal' (Clase 0) como medida de precaución/sensibilidad.
    if top1_confidence < confidence_threshold:
         top1_index = 0 

    # 2. Obtener Score Base y Aplicar Lógica Adaptativa
    base_info = LABELS_RISK_DATA.get(top1_index, LABELS_RISK_DATA[0]) # Default a Normal
    
    final_score = base_info['score']
    final_nivel = base_info['nivel']
    final_impacto = base_info['impacto']
    
    # Lógica de Adaptación (Aplica la penalización/bonificación)
    user_experience = user_profile.get('experiencia', 'Intermedio')
    score_adjustment = RISK_FACTORS.get(user_experience, 0)
    
    if top1_index != 0:
        # Penaliza el score (lo hace más bajo, es decir, más peligroso) si es principiante
        final_score = max(5, final_score - score_adjustment) 
        
        if score_adjustment > 0:
            final_impacto = f"({user_experience}) {final_impacto}. El riesgo es incrementado por el perfil seleccionado."
        elif score_adjustment < 0:
             final_impacto = f"({user_experience}) {final_impacto}. Tu experiencia mitiga ligeramente la severidad."
    
    # 3. Determinar Mensaje General (Usando el score final)
    if final_score >= 80:
        warning_message = "✅ ¡Excelente! Conducción Segura. Mantente concentrado."
    elif final_score >= 40:
        warning_message = "⚠️ Precaución. Riesgo Moderado. Reajuste su postura y atención."
    else:
        warning_message = "🚨 ¡PELIGRO CRÍTICO! Detenga la distracción inmediatamente."
    
    # 4. Mensaje Completo para el Front (con el mensaje general como título)
    full_front_message = f"{warning_message}<br><br>{base_info['impacto']}"

    return {
        "class_name": base_info['name'],
        "confidence": f"{top1_confidence:.2%}",
        "message": full_front_message,
        "message_type": base_info['tipo'], # success, warning, error
        "final_score": final_score,         # Nuevo: Puntuación de 0 a 100
        "risk_level": final_nivel,          # Nuevo: CRÍTICO, ALTO, Moderado, Bajo
        "impact_message": final_impacto,    # Nuevo: Mensaje adaptado
        "used_threshold": confidence_threshold # Nuevo: Umbral usado para la inferencia
    }


# --- Carga del Modelo ---
try:
    MODEL = YOLO("best.pt")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo best.pt. Asegúrate de que el archivo existe. {e}")
    MODEL = None

# --- Rutas de Flask ---

@app.route('/', methods=['GET'])
def index():
    """Muestra la página inicial."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe el archivo, el umbral de confianza y el perfil del usuario, procesa y devuelve JSON."""
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No hay archivo en la solicitud"}), 400
    
    # 1. RECUPERAR DATOS DEL FORMULARIO Y PERFIL (NUEVO)
    # Estos campos son enviados por el formulario HTML
    confidence_threshold = request.form.get('threshold', 0.5) 
    user_experience = request.form.get('user_experience', 'Intermedio') 
    
    # Estructura del perfil
    user_profile = {
        'experiencia': user_experience,
    }

    try:
        # Asegura que el umbral sea un float válido
        confidence_threshold = float(confidence_threshold)
        if not (0.1 <= confidence_threshold <= 0.99): confidence_threshold = 0.5
    except ValueError:
        confidence_threshold = 0.5


    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No hay archivo seleccionado"}), 400

    try:
        # Leer la imagen
        image_stream = io.BytesIO(file.read())
        img = Image.open(image_stream)

        # 2. Realizar la predicción con el umbral del usuario (NUEVO)
        # Se pasa el parámetro 'conf' a la función predict de YOLO
        results = MODEL.predict(img, conf=confidence_threshold, verbose=False)
        
        if results and results[0].probs:
            # 3. Procesar resultados adaptativos
            prediction_data = get_prediction_data(results[0].probs, confidence_threshold, user_profile)
            return jsonify(prediction_data)
        else:
            return jsonify({"error": "Error al procesar la predicción"}), 500

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error en el servidor: {e}"}), 500

if __name__ == '__main__':
    # Usar el puerto 5000 o el que necesites
    app.run(debug=True, host='0.0.0.0', port=5000)
