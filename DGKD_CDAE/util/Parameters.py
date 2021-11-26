import json

# https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py

""" Load hyperparameters from a json file.
Example:
```
params = Params(json_path)
print(params.learning_rate)
params.learning_rate = 0.5  # change the value of learning_rate in params
```
"""


class Parameters(object):
    def __init__(self, json_path, name='Configure'):
        self.name = name
        self.update(json_path)

    def save(self, json_path):
        """Save parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update_from_dict(self, dictionary):
        """Load parameters from dictionary"""
        self.__dict__.update(dictionary)

    def update(self, json_path):
        """Load parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def add_param(self, k, v):
        self.__dict__[k] = v

    @property
    def dict(self):
        """Give dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    def __contains__(self, key):
        return key in self.__dict__

    def __str__(self):
        # return string representation of 'Parameters' class
        # print(Parameters) or str(Parameters)
        ret = '\n=========== [%s] ===========\n' % self.name
        for k in self.__dict__:
            if k != 'grid_search_params':
                ret += '%s: %s\n' % (str(k), str(self.__dict__[k]))
        return ret