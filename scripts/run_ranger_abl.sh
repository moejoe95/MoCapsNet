# number of capsules

echo "ResCapsNet, 64 caps"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --num_caps 64 --residual --epochs 100

echo "MoCapsNet, 64 caps"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --num_caps 64 --residual --momentum --epochs 100


# batch size

echo "ResCapsNet, 64 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --epochs 100 --batch_size 64
echo "MoCapsNet, 64 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --momentum --epochs 100 --batch_size 64

echo "ResCapsNet, 128 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --epochs 100 --batch_size 128
echo "MoCapsNet, 128 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --momentum --epochs 100 --batch_size 128

echo "ResCapsNet, 256 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --epochs 100 --batch_size 256
echo "MoCapsNet, 256 batch size"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --momentum --epochs 100 --batch_size 256


# learning rate
echo "ResCapsNet, 0.1 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.1 --residual --epochs 100 
echo "MoCapsNet, 0.1 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.1 --residual --momentum --epochs 100 
echo "ResCapsNet, 0.01 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --epochs 100 
echo "MoCapsNet, 0.01 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.01 --residual --momentum --epochs 100 
echo "ResCapsNet, 0.001 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.001 --residual --epochs 100 
echo "MoCapsNet, 0.001 learning rate"
python main.py --num_res_block 2 --dataset cifar --optimizer ranger21 --learning_rate 0.001 --residual --momentum --epochs 100 