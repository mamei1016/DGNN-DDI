# mamei-DGNN-DDI
source for paper DGNN-DDI
DGNN-DDI is designed to a dual GNN for drug-drug interaction prediction based on molecular structure  and can provide explanations that are consistent with pharmacologists.
## Note
We have added comments to drugbank/data_preprocessing.py and drugbank/model.py. If you are interested in the technical details of preprocessing steps and algorithms, I think those comments would be helpful. 
## Requirements  
numpy ==1.22.3          
pandas  == 1.4.3           
python  == 3.8.13             
pytorch   == 1.12.0           
rdkit   == 2020.09.1     
scikit-learn  ==1.1.1                   
torch-geometric ==2.0.4                   
torch-scatter == 2.0.9                  
torch-sparse == 0.6.14                 
tqdm  ==   4.64.0  
## Step-by-step running:  

- First,  run data_preprocessing.py using  ' data_preprocessing.py -d drugbank -o all`  
  Running data_preprocessing.py convert the raw data into graph format.

- Second, run train.py using  ' train.py --fold 0 --save_model' 
