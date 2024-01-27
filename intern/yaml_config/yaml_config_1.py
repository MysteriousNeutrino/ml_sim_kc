import yaml


def yaml_to_env(config, parent_key="", result=None):
    if result is None:
        result = []

    if isinstance(config, dict):
        for key, value in config.items():
            new_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                yaml_to_env(value, new_key, result)
            else:
                # Проверяем, содержатся ли в строковом значении пробелы
                if isinstance(value, str) and ' ' in value:
                    result.append(f"{new_key}=\"{value}\"\n")
                else:
                    result.append(f"{new_key}={value}\n")
    else:
        # Проверяем, содержатся ли в строковом значении пробелы
        if isinstance(config, str) and ' ' in config:
            result.append(f"{parent_key}=\"{config}\"\n")
        else:
            result.append(f"{parent_key}={config}\n")

    return "".join(result)


def env_to_yaml(env_text):
    env_list = env_text.split("\n")
    config_dict = {}

    for line in env_list:
        if "=" in line:
            key, value = line.split("=")
            keys = key.split(".")
            current_dict = config_dict

            for k in keys[:-1]:
                current_dict = current_dict.setdefault(k, {})

            # Преобразование строк 'True' и 'False' в булевы значения
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False

            current_dict[keys[-1]] = value

    yaml_str = yaml.dump(config_dict, default_flow_style=False)

    # Убираем кавычки вокруг строк
    yaml_str = yaml_str.replace("'", "")

    return yaml_str


config_text = """
log_dir: "Models/VCTK20"
save_freq: 2
device: "cuda"
epochs: 150
batch_size: 5
pretrained_model: ""
load_only_params: false
fp16_run: true

train_data: "Data/train_list.txt"
val_data: "Data/val_list.txt"

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00100.pth"

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

model_params:
  dim_in: 64
  style_dim: 64
  latent_dim: 16
  num_domains: 20
  max_conv_dim: 512
  n_repeat: 4
  w_hpf: 0
  F0_channel: 256

loss_params:
  g_loss:
    lambda_sty: 1.
    lambda_cyc: 5.
    lambda_ds: 1.
    lambda_norm: 1.
    lambda_asr: 10.
    lambda_f0: 5.
    lambda_f0_sty: 0.1
    lambda_adv: 2.
    lambda_adv_cls: 0.5
    norm_bias: 0.5
  d_loss:
    lambda_reg: 1.
    lambda_adv_cls: 0.1
    lambda_con_reg: 10.

  adv_cls_epoch: 50
  con_reg_epoch: 30

optimizer_params:
  lr: 0.0001
"""

config = yaml.safe_load(config_text)
# print(config)
env_text = yaml_to_env(config)
print(env_text)

# new_config_text = env_to_yaml(env_text)
# print("\nConverted back to YAML:")
# print(new_config_text)
