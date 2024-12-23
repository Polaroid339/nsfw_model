import os
from nsfw_detector import predict

# Carregar o modelo
model = predict.load_model('./nsfw_mobilenet2.224x224.h5')  # Ajuste o caminho, se necessário

def classify_images_in_directory(model, directory):
    results = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Verifica se é uma imagem
            try:
                result = predict.classify(model, file_path)
                results[filename] = result[file_path]
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
    return results

# Diretório com as imagens
image_directory = './images'  # Substitua pelo caminho da sua pasta

# Classificar imagens no diretório
all_results = classify_images_in_directory(model, image_directory)

# Exibir resultados formatados
print("Resultados da classificação:")
for image_name, predictions in all_results.items():
    print(f"\nImagem: {image_name}")
    for label, probability in predictions.items():
        print(f"  {label}: {probability * 100:.2f}%")
