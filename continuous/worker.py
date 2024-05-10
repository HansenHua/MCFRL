from util import *

class Context(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Context, self).__init__()
        self.context = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    
    def forward(self, buffer):
        return self.context(buffer)

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, num):
        super(Attention, self).__init__()
        self.num = num
        self.enc_list = []
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        for _ in range(self.num):
            self.enc_list.append(nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ))
    
    def forward(self, state, context):
        state = torch.from_numpy(state).float()
        encoded_state = []
        alpha = []
        for i in range(self.num):
            encoded_state.append(self.enc_list[i](state))
        for i in range(self.num):
            alpha.append(torch.sum(encoded_state[i] * context, dim=0))
        alpha = F.softmax(torch.tensor(alpha), dim=0).unsqueeze(-1)
        alpha = alpha.repeat(1, 1, 64).squeeze(0)
        aggregated_state = torch.sum(alpha * torch.stack(encoded_state), dim=0)
        aggregated_state = self.mlp(aggregated_state)

        return aggregated_state

class Encoder(nn.Module):
    def __init__(self, in_dim, in_dim_c, attention_dim, context_dim, num_att):
        super(Encoder, self).__init__()
        self.context = Context(in_dim_c, context_dim)
        self.attention = Attention(in_dim, attention_dim, num_att)
    
    def forward(self, state, buffer):
        c = self.context(buffer)
        c_nograd = copy.deepcopy(c.detach())
        enc = self.attention(state, c_nograd)
        return torch.cat([c, enc], dim=0)

class policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(policy, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.network = nn.Sequential(
                init_(nn.Linear(in_dim, 256)),
                nn.Tanh(),
                init_(nn.Linear(256, 256)),
                nn.Tanh(),
            )
        self.output = init_(nn.Linear(256, out_dim))
        self.output_ = init_(nn.Linear(256, out_dim))
    
    def forward(self, state):
        # state = torch.from_numpy(state).float()
        s = self.network(state)
        mu = self.output(s)
        sigma = self.output_(s)

        return mu, sigma

class critic(nn.Module):
    def __init__(self, in_dim):
        super(critic, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.network = nn.Sequential(
                init_(nn.Linear(in_dim, 256)),
                nn.ReLU(),
                init_(nn.Linear(256, 256)),
                nn.ReLU(),
                init_(nn.Linear(256, 1))
            )
    
    def forward(self, x):
        c = self.network(x)
        return c

class Worker(nn.Module):
    def __init__(self, env):
        super(Worker, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.env = env
        self.learning_rate_a = self.method_conf['learning_rate_a']
        self.learning_rate_c = self.method_conf['learning_rate_c']
        self.attention_dim = self.method_conf['attention_dim']
        self.context_dim = self.method_conf['context_dim']
        self.num_attention = self.method_conf['num_attention']
        self.max_step = 1000
        self.gamma = self.method_conf['gamma']
        self.tau = self.method_conf['tau']
        self.c = self.method_conf['c']
        self.buffer = torch.zeros(2*self.env.observation_space.shape[0]+self.env.action_space.shape[0])
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.encoder = Encoder(self.observation_space, 2*self.observation_space+self.action_space, self.attention_dim, self.context_dim, self.num_attention)
        self.network = policy(self.attention_dim+self.context_dim, self.action_space)
        self.old_network = policy(self.attention_dim+self.context_dim, self.action_space)
        self.critic = critic(self.attention_dim+self.context_dim)
        self.optimizer_new = torch.optim.Adam(self.network.network.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_old = torch.optim.Adam(self.old_network.network.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_c, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_encoder = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(self.optimizer_new, self.method_conf['decay_rate'])
        self.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(self.optimizer_old, self.method_conf['decay_rate'])
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(self.optimizer_critic, self.method_conf['decay_rate'])
        self.pi = Variable(torch.FloatTensor([math.pi]))

    def normal(self, x, mu, sigma_sq):
        a = ( -1 * (Variable(x)-mu).pow(2) / (2*sigma_sq) ).exp()
        b = 1 / ( 2 * sigma_sq * self.pi.expand_as(sigma_sq) ).sqrt()
        return a*b
    
    def gen_action(self, state):
        # state = torch.from_numpy(state).float()
        mu, sigma = self.network(state)
        sigma = F.softplus(sigma)

        eps = torch.randn(mu.size())

        action = (mu + sigma.sqrt()*Variable(eps)).clamp(-self.max_action, self.max_action).data
        prob = self.normal(action, mu, sigma)

        log_prob = prob.log()
        return action, log_prob

    def gen_action_prob(self, state, action):
        # state = torch.from_numpy(state).float()
        mu, sigma = self.old_network(state)
        sigma = F.softplus(sigma)
        prob = self.normal(action, mu, sigma)
        log_prob = prob.log()
        
        return log_prob
    
    def gen_critic(self, state):
        # state = torch.from_numpy(state).float()
        v = self.critic(state)

        return v
    
    def collect_trajectory(self, batch_size):
        state_batch = []
        state_encoded_batch = []
        action_batch = []
        action_prob_batch = []
        batch_weights = []
        critic_batch = []
        state_prime_batch = []
        r_batch = []
        for _ in range(batch_size):
            state, reward, done = self.env.reset(), 0, False
            prior_s_a = torch.zeros(self.observation_space+self.action_space)
            self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
            state_encoded = self.encoder(state, self.buffer)
            reward_batch = []
            step = 0
            while True:
                step += 1
                state_encoded = self.encoder(state, self.buffer)
                action, action_prob = self.gen_action(state_encoded)
                prior_s_a = torch.cat([torch.tensor(state), action], dim=0)

                state_batch.append(state)
                state_encoded_batch.append(state_encoded)
                v = self.gen_critic(state_encoded)
                critic_batch.append(v)
                state, reward, done, _ = self.env.step(action.numpy())
                self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
                reward_batch.append(reward)
                r_batch.append(reward)
                action_batch.append(action)
                state_prime_batch.append(state)
                action_prob_batch.append(action_prob)

                if done:
                    returns = []
                    gae = 0
                    for i in reversed(range(len(reward_batch))):
                        if i == len(reward_batch) - 1:
                            delta = reward_batch[i] - critic_batch[i]
                        else:
                            delta = r_batch[i] + self.gamma * critic_batch[i+1] - critic_batch[i]
                        gae = delta + self.gamma * self.tau * gae
                        returns.insert(0, gae + critic_batch[i])
                    returns = torch.tensor(returns, dtype=torch.float32)
                
                    batch_weights += returns
                    break
        
        batch_weights = torch.as_tensor(batch_weights, dtype = torch.float32)

        state_prime_batch = torch.as_tensor(state_prime_batch, dtype = torch.float32)
        r_batch = torch.as_tensor(r_batch, dtype = torch.float32)

        return batch_weights, state_batch, action_batch, action_prob_batch, critic_batch, state_prime_batch, r_batch, state_encoded_batch
    
    def train(self, batch_size, step):
        returns, state_batch, action_batch, action_prob_batch, critic_batch, _, _, state_encoded_batch = self.collect_trajectory(batch_size)

        returns = returns.unsqueeze(-1).repeat(1, self.action_space)
        advantage = returns - torch.stack(critic_batch).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-20)
        
        grad = [item.grad for item in self.network.parameters()]

        self.optimizer_new.zero_grad()
        self.optimizer_encoder.zero_grad()
        action_loss = -(torch.stack(action_prob_batch) * advantage).mean()
        action_loss.backward()

        old_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob(state_encoded_batch[idx].detach(), action_batch[idx])
            old_logp.append(action_prob)
        old_logp = torch.stack(old_logp)

        ratios = torch.exp(old_logp.detach() - torch.stack(action_prob_batch).detach())

        
        loss_old = -(old_logp * advantage * ratios).mean()
        self.optimizer_old.zero_grad()
        loss_old.backward()

        grad_old = [item.grad for item in self.old_network.parameters()]

        if grad[0] is not None:
            for idx,item in enumerate(self.network.parameters()):
                item.grad = item.grad + (1 - self.c * self.learning_rate_a**2) * (grad[idx] - grad_old[idx])
        
        grad = [item.grad for item in self.network.parameters()]

        self.old_model = copy.deepcopy(self.network)

        self.optimizer_new.step()
        self.optimizer_encoder.step()

        if step > self.method_conf['decay_start_iter_id']:
            self.lr_scheduler_new.step()
            self.lr_scheduler_old.step()
            self.lr_scheduler_critic.step()
    
    def test(self, i):
        if self.method_conf['mode'] == 'homo':
            file = open('report.txt', 'a')
            sum_r = 0
            for _ in range(10):
                state = self.env.reset()
                prior_s_a = torch.zeros(self.observation_space+self.action_space)
                self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
                state_encoded = self.encoder(state, self.buffer)
                while True:
                    action, _ = self.gen_action(state_encoded)
                    prior_s_a = torch.cat([torch.tensor(state), action], dim=0)
                    state, reward, done, _ = self.env.step(action.numpy())
                    self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
                    state_encoded = self.encoder(state, self.buffer)
                    sum_r += reward
                    if done:
                        break
            file.write(str(i) + ':' + str(sum_r / 10) +'\n')
            print('worker test', sum_r / 10)
            return sum_r / 10
        else:
            file = open('report.txt', 'a')
            sum_r = 0
            env_list = []
            if self.env_name_list[0] == 'cartpole':
                from env import CartPoleEnv
                for i in range(self.num_worker):
                    env = CartPoleEnv(1+0.2*i)
                    env_list.append(env)
            elif self.env_name_list[0] == 'halfcheetah':
                for i in range(self.num_worker):
                    env = gym.make('HalfCheetah-v2',xml_file='./xml/halfcheetah/halfcheetah_'+str(i)+'.xml')
                    env_list.append(env)
            for env in env_list:
                for _ in range(10):
                    state = self.env.reset()
                    prior_s_a = torch.zeros(self.observation_space+self.action_space)
                    self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
                    state_encoded = self.encoder(state, self.buffer)
                    while True:
                        action, _ = self.gen_action(state_encoded)
                        prior_s_a = torch.cat([torch.tensor(state), action], dim=0)
                        state, reward, done, _ = self.env.step(action.numpy())
                        self.buffer = torch.cat([prior_s_a, torch.tensor(state)],dim=0)
                        state_encoded = self.encoder(state, self.buffer)
                        sum_r += reward
                        if done:
                            break
            file.write(str(i) + ':' + str(sum_r / (10*len(env_list))) +'\n')
            print('worker test', sum_r / (10*len(env_list)))
            return sum_r / (10*len(env_list))