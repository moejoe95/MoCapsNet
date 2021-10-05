
echo "mnist"

echo "CapsNet"
python main.py --num_res_blocks 2 --dataset mnist --epochs 30 --routing sda
echo "ResCapsNet"
python main.py --num_res_blocks 2 --dataset mnist --epochs 30 --residual --routing sda
echo "MoCapsNet"
python main.py --num_res_blocks 2 --dataset mnist --epochs 30 --residual --momentum --routing sda

echo "cifar"

echo "CapsNet"
python main.py --num_res_blocks 2 --dataset cifar --epochs 60 --routing sda
echo "ResCapsNet"
python main.py --num_res_blocks 2 --dataset cifar --epochs 60 --residual --routing sda
echo "MoCapsNet"
python main.py --num_res_blocks 2 --dataset cifar --epochs 60 --residual --momentum --routing sda

echo "svhn"

echo "CapsNet"
python main.py --num_res_blocks 2 --dataset svhn --epochs 60 --routing sda
echo "ResCapsNet"
python main.py --num_res_blocks 2 --dataset svhn --epochs 60 --residual --routing sda
echo "MoCapsNet"
python main.py --num_res_blocks 2 --dataset svhn --epochs 60 --residual --momentum --routing sda
