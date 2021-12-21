import yaml

def reader(ymlFile):
    with open(ymlFile) as file:
        data = yaml.full_load(file)

    return data