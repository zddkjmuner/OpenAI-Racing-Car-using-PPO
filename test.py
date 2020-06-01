import numpy as np

import gym
import torch
import torch.nn as nn

import numpy as np
import cv2
import matplotlib.pyplot as plt




NUM_IMG_STACK = 4
NUM_ACTION_REPEAT = 8
SEED = 0


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed(SEED)



class TEST_Env():

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(SEED)

    @staticmethod
    def store_reward():
        count = 0
        L = 100
        buffer = np.zeros(L)

        def store(reward):
            nonlocal count
            buffer[count] = reward
            count = (count + 1) % L
            mean_buffer = np.mean(buffer)
            return mean_buffer

        return store


    def reset(self):
        self.count = 0
        self.av_r = self.store_reward()

        self.fail = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * NUM_IMG_STACK
        np_img_stack = np.array(self.stack)
        return np_img_stack

    def step(self, action):
        cum_reward = 0
        for _ in range(NUM_ACTION_REPEAT):
            img_rgb, reward, fail, _ = self.env.step(action)
            # don't penalize "die state"
            if fail:
                reward += 100

            # green penalty
            avg = np.mean(img_rgb[:, :, 1])
            if avg > 185.0:
                reward -= 0.05
            cum_reward += reward
            # if no reward recently, end the episode
            if self.av_r(reward) > -0.1:
                finish = False
            else:
                finish = True

            # finish = True if self.av_r(reward) <= -0.1 else False
            if finish or fail:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        L_img = len(self.stack)
        assert L_img == NUM_IMG_STACK
        return np.array(self.stack), cum_reward, finish, fail

    def render(self):
        self.env.render()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

class ACNet(nn.Module):
    def __init__(self):
        super(ACNet, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2,),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.v_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softplus()
        )
        self.apply(self._weights_init)

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        x = self.cnn_layer6(x)
        x = x.view(-1, 256)
        v = self.v_head(x)
        alpha = self.alpha_head(self.fc_layer(x)) + 1
        beta = self.beta_head(self.fc_layer(x)) + 1
        return (alpha, beta), v

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
        # self.CNN = nn.Sequential( 
            nn.Conv2d(NUM_IMG_STACK, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        # x = self.CNN(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_params.pth', map_location=lambda storage, loc: storage))


if __name__ == "__main__":
    agent = Agent()
    agent.load_param()
    env = TEST_Env()

    # env_video = gym.make('CarRacing-v0')
    # video = VideoRecorder(env_video, path='./car.mp4', enabled=True)

    # training_records = []
    testing_rw_list = []
    # running_score = 0
    state = env.reset()
    NUM_EPs = 10
    for i in range(NUM_EPs):
        val = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            cvt_action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            nxt_state, reward, finish, fail = env.step(cvt_action)
            # if args.render:
            env.render()
            val += reward
            state = nxt_state
            if finish or fail:
                break
        testing_rw_list.append(val)
        # print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        print("Epoch:{}/{}, Cumulating_Reward:{}".format(i, NUM_EPs, val))

    # print("Average Testing Score is: {}".format(sum(testing_rw_list)/len(testing_rw_list)))
    plt.figure()
    plt.plot(range(len(testing_rw_list)), testing_rw_list)
    plt.xlabel("Number of Epoches")
    plt.ylabel("Cumulating Reward")
    plt.title("Cumulating Reward VS. Iteration Steps")
    plt.show()
    # print("Average Testing Score is: {}".format(sum(testing_rw_list)/len(testing_rw_list)))
    # p = ImageGrab.grab()#获得当前屏幕
    # k=np.zeros((200,200),np.uint8)
    # a,b=p.size#获得当前屏幕的大小
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')#编码格式
    # video = cv2.VideoWriter('test.avi', fourcc, 16, (a, b))#输出文件命名为test.mp4,帧率为16，可以自己设置
    # while True:
    #     im = ImageGrab.grab()
    #     imm=cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)#转为opencv的BGR格式
    #     video.write(imm)
    #     cv2.imshow('imm', k)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # video.release()
    # cv2.destroyAllWindows()