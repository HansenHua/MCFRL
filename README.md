# MCFRL
This work "Momentum-Based Contextual Federated Reinforcement Learning" has been submitted in ToN.
## :page_facing_up: Description
we propose a new FRL method, \emph{Momentum-based Contextual Federated Reinforcement Learning} (), capable of jointly optimizing both the interaction and communication complexities while coping with the environment or task heterogeneities.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- [MuJoCo == 2.3.6](http://www.mujoco.org) 
- NVIDIA GPU (RTX A6000) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone [https://github.com/HansenHua/MCFRL.git](https://github.com/HansenHua/MCFRL.git)
    cd code
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```
Then the usage information will be shown as following
```
usage: main.py [-h] [--env_name ENV_NAME] [--method METHOD] [--gamma GAMMA] [--batch_size BATCH_SIZE]
               [--local_update LOCAL_UPDATE] [--num_worker NUM_WORKER] [--average_type AVERAGE_TYPE] [--c C]
               [--seed SEED] [--lr_a LR_A] [--lr_c LR_C]
               mode max_iteration

positional arguments:
  mode                  train or test
  max_iteration         maximum training iteration

optional arguments:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   the name of environment
  --method METHOD       method name
  --gamma GAMMA         gamma
  --batch_size BATCH_SIZE
                        batch_size
  --local_update LOCAL_UPDATE
                        frequency of local update
  --num_worker NUM_WORKER
                        number of federated agents
  --average_type AVERAGE_TYPE
                        average type (target/network/critic)
  --c C                 momentum parameter
  --seed SEED           random seed
  --lr_a LR_A           learning rate of actor
  --lr_c LR_C           learning rate of critic
```
Test the trained models provided in [MCFRL](https://github.com/HansenHua/MCFRL/tree/main/log).
```
python main.py CartPole-v1 MCFRL test
```
## :computer: Training

We provide complete training codes for MCFRL.<br>
You could adapt it to your own needs.

	```
    python main.py CartPole-v1 MCFRL train
	```

## :checkered_flag: Testing
1. Testing
	```
	python main.py CartPole-v1  test
	```
2. Illustration

We alse provide the performance of our model. The illustration videos are stored in [MCFRL-Online-Federated-Reinforcement-Learning/performance](https://github.com/HansenHua/MCFRL/tree/main/performance).

## :e-mail: Contact

If you have any question, please email `xingyuanhua@bit.edu.cn`.
