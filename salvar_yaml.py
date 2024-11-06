import yaml

data = {
    'nome': 'Maria',
    'idade': 25,
    'cidade': 'Rio de Janeiro'
}

# Salvar dados em um arquivo YAML
with open('novo_dados.yaml', 'w') as file:
    yaml.dump(data, file)