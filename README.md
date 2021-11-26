# Dually Guided Knowledge Distillation (DGKD)
![FFF](https://user-images.githubusercontent.com/56680371/143531372-beadf494-1c59-4d06-b5d1-b0a0ec639059.PNG)

## Requirements
Tensorflow 2.6


## Rapid usage
### Training teacher \#2
1. Set ./main_config.json<br/>
    model_name: CDAE<br/>
    <br/>
2. Set ./model_config/CDAE.json  <br/>
    hidden_dim: 100<br/>
    save_output: true<br/>
   <br/>
3. Run main.py<br/>
    <br/>
4. Teacher model is saved to ./data/"data_name"/"model_name".p<br/>

### Knowledge Distillation (Teacher model training required)
1. Set ./main_config.json<br/>
    model_name: CDAE_DGKD<br/>
    <br/>
2. Set ./model_config/CDAE_DGKD<br/>
    teacher_dim: 100<br/>
    <br/>
3. Set ./model/CDAE_DGKD.py<br/>
    self.unintFilePath: path for uninteresting file (file containing the result of teacher \#1)<br/>
    self.intFilePath: path for interesting file (file containing the result of teacher \#1)<br/>
    * The data files for teacher \#1's result can be found at https://drive.google.com/drive/folders/11apeRr60DAZBHZHEz5zUhZCAWZZW_8F4?usp=sharing
    <br/>
4. Run main.py<br/>


## Hyper-parameter settings
Set ./model_config/"model_name".json<br/>


## Experiment Environment Settings
Set ./model_config.json<br/>
    ex) data<br/>
      dataset: "ml1m"<br/>

