# CIFAR

# 15 blocks
python main.py --residual --momentum --num_res_blocks 15 --dataset cifar --epochs 60
# 25 blocks
python main.py --residual --momentum --num_res_blocks 25 --dataset cifar --epochs 60

# SVHN

# 15 blocks
python main.py --residual --momentum --num_res_blocks 15 --dataset svhn --epochs 60
# 25 blocks
python main.py --residual --momentum --num_res_blocks 25 --dataset svhn --epochs 60

# MNIST

# 15 blocks
python main.py --residual --momentum --num_res_blocks 15 --dataset mnist --epochs 60
# 25 blocks
python main.py --residual --momentum --num_res_blocks 25 --dataset mnist --epochs 60