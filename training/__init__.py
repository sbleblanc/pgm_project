import torch
import os

class State(object):
    def __init__(self, state_dict, others):
        self.__state_dict = state_dict
        self.__others = others
        
    def get_data(self):
        return {
            **{k:v.state_dict() for k, v in self.__state_dict.items()},
            **self.__others
        }
        
    def save(self, path):
        data = self.get_data()
        torch.save(data, path)
        return self
        
    def load(self, path):
        if not os.path.isfile(path): return
        data = torch.load(path)
        for k,v in self.__state_dict.items():
            v.load_state_dict(data[k])
        for k,v in self.__others.items():
            self. __others[k] = data[k]
        return self
            
    def __getitem__(self, k):
        if k in self.__state_dict:
            return self.__state_dict[k]
        return self.__others[k]
    
    def __setitem__(self, k, v):
        if k in self.__state_dict:
            self.__state_dict[k] = v
        elif k in self.__others:
            self.__others[k] = v
        else:
            raise KeyError('key not found: {}'.format(k))

