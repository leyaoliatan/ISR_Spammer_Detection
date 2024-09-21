#use 5% data
python ./ISR/src/main.py --train_size=0.05 --dataset=amazon --model=GIN_adv --strategy=uncertainty --hidden=32 --save_pred=True