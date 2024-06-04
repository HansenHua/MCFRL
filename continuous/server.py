from util import *
from worker import Worker

class Server:
    def __init__(self, method):
        self.method_conf = get_global_dict_value('method_conf')
        self.env_name_list = self.method_conf['env_name']
        self.method = method
        self.c = self.method_conf['c']
        self.worker_list = []
        self.num_worker = self.method_conf['num_worker']
        self.cur_step = 0
        self.batch_size = self.method_conf['batch_size']
        if self.method_conf['mode'] == 'hete':
            if self.env_name_list == 'CartPole-v1':
                from env import CartPoleEnv
                for i in range(self.num_worker):
                    env = CartPoleEnv(1+0.2*i)
                    self.worker_list.append(Worker(env))
            elif self.env_name_list == 'HalfCheetah-v2':
                for i in range(self.num_worker):
                    env = gym.make('HalfCheetah-v2',xml_file='./xml/halfcheetah/halfcheetah_'+str(i)+'.xml')
                    self.worker_list.append(Worker(env))
            elif self.env_name_list == 'metaworld-10':
                import metaworld
                mt10 = metaworld.MT10()
                for name, env_cls in mt10.train_classes.items():
                    env = env_cls()
                    task = random.choice([task for task in mt10.train_tasks
                            if task.env_name == name])
                    env.set_task(task)
                    self.worker_list.append(Worker(env))
            else:
                import metaworld
                mt50 = metaworld.MT50()
                for name, env_cls in mt50.train_classes.items():
                    env = env_cls()
                    task = random.choice([task for task in mt50.train_tasks
                            if task.env_name == name])
                    env.set_task(task)
                    self.worker_list.append(Worker(env))
        else:
            for i in range(self.num_worker):
                env = gym.make(self.env_name_list)
                self.worker_list.append(Worker(env))
        
        self.master = copy.deepcopy(self.worker_list[0])
    
    def share_model(self, step):        
        self.cur_step = step
        for w in self.worker_list:
            w.train(self.batch_size, step)
        
        local_weights_critic = []
        local_weights_network = []
        local_weights_encoder = []
        for w in self.worker_list:
            local_weights_network.append(copy.deepcopy(w.network.state_dict()))
            local_weights_critic.append(copy.deepcopy(w.critic.state_dict()))
            local_weights_encoder.append(copy.deepcopy(w.encoder.state_dict()))

        global_weights_network = self.average_weights(local_weights_network)
        global_weights_critic = self.average_weights(local_weights_critic)
        global_weights_encoder = self.average_weights(local_weights_encoder)

        self.master.network.load_state_dict(global_weights_network)
        self.master.critic.load_state_dict(global_weights_critic)
        self.master.encoder.load_state_dict(global_weights_encoder)
        
        for w in self.worker_list:
            # w.critic = copy.deepcopy(self.master.critic)
            w.network = copy.deepcopy(self.master.network)
            w.encoder = copy.deepcopy(self.master.encoder)

    def train(self, step):
        self.cur_step = step
        g = []
        gradient = []
        for w in self.worker_list:
            grad = w.train(self.batch_size, step)
            gradient.append(grad)


    def test(self, ):
        score = self.master.test(self.cur_step)
    
    def average_weights(self, w):
        set = []
        for i in range(self.num_worker):
            set.append(i)
        w_avg = copy.deepcopy(w[set[0]])
        for key in w_avg.keys():
            for i in set:
                if i == set[0]:
                    continue
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

