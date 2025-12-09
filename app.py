from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import pandas as pd # Aunque pandas ya no se usa, lo mantengo por si acaso, pero podría quitarse.

app = Flask(__name__)

# --- 1. Definir las Etiquetas y Mensajes ---
LABELS = {
    0: 'Conducción Normal',
    1: 'Enviando mensajes de texto (derecha)',
    2: 'Hablando por teléfono (derecha)',
    3: 'Enviando mensajes de texto (izquierda)',
    4: 'Hablando por teléfono (izquierda)',
    5: 'Operando la radio/climatizador',
    6: 'Bebiendo',
    7: 'Alcanzando algo detrás',
    8: 'Aseo personal (Maquillaje/Pelo)',
    9: 'Hablando con el pasajero'
}

def get_prediction_data(probs):
    """Procesa los resultados de YOLOv8 y los prepara para el frontend."""
    # Obtenemos la clase principal y su confianza
    top1_index = probs.top1
    top1_confidence = probs.top1conf.item()
    predicted_class_name = LABELS.get(top1_index, "Desconocida")
    
    # Mensaje de advertencia
    DANGEROUS_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]
    if top1_index == 0:
        warning_message = "✅ ¡Excelente! Conducción segura."
        warning_type = "success"
    elif top1_index in DANGEROUS_CLASSES:
        warning_message = f"🚨 ¡PELIGRO! Concéntrate en la vía."
        warning_type = "error"
    else:
        warning_message = "⚠️ Advertencia. Mantén la atención."
        warning_type = "warning"
        
    # **SE ELIMINÓ LA PARTE DE PROBABILIDADES DETALLADAS (prob_map)**
    
    return {
        "class_name": predicted_class_name,
        "confidence": f"{top1_confidence:.2%}",
        "message": warning_message,
        "message_type": warning_type,
        # **SE ELIMINÓ 'probabilities' DEL DICT DE RESPUESTA**
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
    """Recibe el archivo, lo procesa con el modelo y devuelve JSON."""
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No hay archivo en la solicitud"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No hay archivo seleccionado"}), 400

    try:
        # Leer la imagen desde el stream
        image_stream = io.BytesIO(file.read())
        img = Image.open(image_stream)

        # Realizar la predicción
        results = MODEL.predict(img, verbose=False)
        
        if results and results[0].probs:
            prediction_data = get_prediction_data(results[0].probs)
            return jsonify(prediction_data)
        else:
            return jsonify({"error": "Error al procesar la predicción"}), 500

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error en el servidor: {e}"}), 500

if __name__ == '__main__':
    # Usar debug=True solo en desarrollo
    app.run(debug=True, host='0.0.0.0', port=5000)