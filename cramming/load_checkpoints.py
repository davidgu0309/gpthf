import torch
import os

to_load = ['llama-standard']
#['opt-125m', 'opt-350m', 'llama-small', 'llama-standard']
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs')


def load_checkpoint(name):
    for i in range(1, 11):
        path = f'{base_path}/{name}/{i}.pth'
        try:
            save_state = torch.load(path, map_location=torch.device("cpu"))
            tmp = save_state["model"]  # why does this end up on GPU?
            tmp = save_state["optim"]
            tmp = save_state["scheduler"]
            tmp = save_state["scaler"]
            tmp = save_state['metadata']
            print(f'{path} is ok' )
        except FileNotFoundError:
            print(f'{path} not found')
        except Exception as e:
            print(f'{path} is not loadable')
            print(e)


for name in to_load:
    load_checkpoint(name)