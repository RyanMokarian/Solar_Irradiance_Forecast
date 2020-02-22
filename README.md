# solar-irradiance
Project 1 of course IFT6759 - Advanced projects in machine learning

## How to test the evaluator script
Check that your configuration is correct in the file eval_user_cfg.json and run the command :
```
python evaluator.py output.txt dummy_test_cfg.json -u="eval_user_cfg.json"
```

## How to Visualize Training with Tensorboard
Open an ssh session :
```
ssh -L 16006:127.0.0.1:6006 guest@helios3.calculquebec.ca
```
Run tensorboard : 
```
tensorboard --logdir=solar-irradiance/logs
```
Open the following url in your browser : 
```
http://127.0.0.1:16006
```
