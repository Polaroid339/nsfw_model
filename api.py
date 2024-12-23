from flask import Flask, request, jsonify
from nsfw_detector import predict
import os

app = Flask(__name__)

# Carregar o modelo
MODEL_PATH = './nsfw_mobilenet2.224x224.h5'
model = predict.load_model(MODEL_PATH)

# Endpoint para classificar uma única imagem
@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'Nenhuma imagem selecionada'}), 400

    # Salvar a imagem temporariamente
    temp_path = './temp_image.jpg'
    image.save(temp_path)

    # Classificar a imagem
    try:
        result = predict.classify(model, temp_path)
        os.remove(temp_path)  # Remover a imagem temporária
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint para classificar imagens em um diretório
@app.route('/classify-directory', methods=['POST'])
def classify_directory():
    if 'directory' not in request.json:
        return jsonify({'error': 'Nenhum diretório fornecido'}), 400

    directory = request.json['directory']
    if not os.path.exists(directory):
        return jsonify({'error': 'O diretório não existe'}), 400

    try:
        results = predict.classify(model, directory)
        formatted_results = {
            image: {label: f"{probability * 100:.2f}%" for label, probability in predictions.items()}
            for image, predictions in results.items()
        }
        return jsonify({'results': formatted_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota de teste
@app.route('/')
def index():
    return jsonify({'message': 'API de Moderação NSFW está funcionando!'})

# Rodar o servidor
if __name__ == '__main__':
    app.run(debug=True)
