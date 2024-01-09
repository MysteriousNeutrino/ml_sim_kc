import yaml


def yaml_to_env(config_file: str) -> str:
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
    processed_data = process_yaml(data)

    return '\n'.join(processed_data)


def process_yaml(data, prefix=""):
    results = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            results.extend(process_yaml(value, new_prefix))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_prefix = f"{prefix}[{index}]"
            results.extend(process_yaml(item, new_prefix))
    else:
        results.append(f"{prefix}.{data}")

    return results


def env_to_yaml(env_list: str) -> str:
    pass

# config_file = r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\intern\yaml_config\example.yaml"
# print(yaml_to_env(config_file))
