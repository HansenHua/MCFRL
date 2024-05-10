from util import *

METHOD_CONF = {
    'mode': 'homo',
    'learning_rate_a': 1e-4,
    'learning_rate_c': 1e-4,
    'gamma': 0.99,
    'eps': 1e-5,
    'env_name': ['Pendulum-v1'],
    'num_worker': 10,
    'batch_size': 20,
    'local_update': 20,
    'c': 3,
    'decay_rate': 0.99,
    'decay_start_iter_id': 500,
    'num_attention':6,
    'attention_dim':64,
    'context_dim':64,
    'tau':0.95
}
