# To find hyperparameters:
batches = [4,8,16,32]
epochs = [50,100,150,200]
freezes = [10,15]
learnings = [0.01,0.001,0.0001]
learnings_final = [0.01,0.001] # on the last epoch lr*lf
rest_params = './additions/params.yaml'