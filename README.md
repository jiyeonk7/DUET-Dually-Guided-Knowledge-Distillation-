# Dually Guided Knowledge Distillation (DGKD)

## Requirements
tensorflow 2.6


## Rapid usage
# Training teacher \#2 & student models
1. Set ./main_config.json
    model_name: CDAE
    
2. Set ./model_config/CDAE.json
    hidden_dim: 100 (teacher) or 10 (student)
    save_output: true (teacher) or false (stduent)
    
-- Teacher model is saved to ./data/"data_name"/"model_name".p

3. Run main.py 


# Knowledge Distillation (Teacher model training required)
1. Set ./main_config.json
    model_name: CDAE_DGKD
    
2. Set ./model_config/CDAE_DGKD
    teacher_dim: 100

3. Set ./model/CDAE_DGKD.py
    self.unFilePath: path for uninteresting file (as a result of teacher \#1)
    self.intFilePath: path for interesting file (as a result of teacher \#1)
    
4. Run main.py


## Hyper-parameter settings
Set ./model_config/"model_name".json


## Experiment Environment Settings
Set ./model_config.json
  ex) data
      dataset: "ml1m
