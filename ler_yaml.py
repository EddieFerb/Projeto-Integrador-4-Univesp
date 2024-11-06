import yaml

# Carregar dados de um arquivo YAML
with open('dados.yaml', 'r') as file:
    data = yaml.safe_load(file)

print(data)
