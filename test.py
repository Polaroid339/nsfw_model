from nsfw_detector import predict

# Carregar o modelo
model = predict.load_model('./nsfw_mobilenet2.224x224.h5')  # Ajuste o caminho, se necessário

# Diretório com as imagens
image_directory = './images'  # Substitua pelo caminho da sua pasta

# Classificar todas as imagens no diretório
results = predict.classify(model, image_directory)

# Exibir os resultados
print("Resultados da classificação (em porcentagem):")
for image_name, predictions in results.items():
    print(f"\nImagem: {image_name}")
    for label, probability in predictions.items():
        print(f"  {label}: {probability * 100:.2f}%")
