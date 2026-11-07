import torch

# Cargar el checkpoint
ruta = "resultados/cnn/modelo_mejor.pt"
checkpoint = torch.load(ruta, map_location="cpu", weights_only=True)

# Mostrar el contenido
print("Claves del checkpoint:", checkpoint.keys())

# Extraer los pesos
pesos = checkpoint["state_dict"]

# Mostrar información
print("\nNúmero de parámetros almacenados:", len(pesos))

for nombre, tensor in pesos.items():
    print(f"{nombre}: {tuple(tensor.shape)}")

# Mostrar configuración y clases
print("\nConfiguración:", checkpoint["cfg"])
print("Clases:", checkpoint["clases"])