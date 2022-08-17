# MNIST 20
python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.001

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001 --routing sda

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001 --lr_decay 0.98

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001 --lr_decay 0.94

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001 --batch_size 256

python main.py --residual --momentum --num_res_blocks 20 --dataset mnist --epochs 30 --learning_rate 0.0001 --batch_size 128

