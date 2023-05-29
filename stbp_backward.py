import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

step = 20
dt = 12
simwin = dt * step
a = 0.5
aa = 0.25 # a /2
Vth = 0.3#0.3
tau = 0.3#0.3

def adjust_learning_rate(lr, optimizer, epoch):
    new_lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class SpikeAct(Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u - Vth > 0 then output = 1
        output = torch.gt(input, 0) 
        return output.float()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # input = u - Vth
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        #hu = abs(input) < a/2 
        #hu = hu.float() / a
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        # print('hu',hu)
        # print('input', input)
        # print(grad_input * hu)
        # print(grad_output)

        return grad_input * hu


spikeAct = SpikeAct.apply


# n1 means n+1
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1 - Vth)
    return u_t1_n1, o_t1_n1


class SpikeNN(nn.Module):
    def __init__(self):
        super(SpikeNN, self).__init__()
        # self.fc2 = nn.Linear(5,3)
        self.fc3 = nn.Linear(30, 2 , bias=None)

        self.device = torch.device("cpu") # or cpu

    def forward(self, input):
        # temp variable define / initial state should be zero
        # fc3_u = fc3_out = spike_sum = torch.zeros(input.shape[0], 3, device=self.device)
        fc3_u = fc3_out = spike_sum = torch.zeros(input.shape[0], 2, device=self.device)

        # fc2_out_list = []
        fc3_out_list = []
        fc3_u_list = []

        for t in range(step):
            in_t = input[ :, :, t]
            # encoding layer

            # fc2_u, fc2_out = state_update(fc2_u, fc2_out, self.fc2(in_t))
            fc3_u, fc3_out = state_update(fc3_u, fc3_out, self.fc3(in_t))
            spike_sum += fc3_out
            # spike_sum += fc2_out
            # fc2_out_list.append(fc2_out)
            fc3_out_list.append(fc3_out)
            fc3_u_list.append(fc3_u)

        return spike_sum / step , fc3_u_list ,   fc3_out_list # rate coding

def f(o):
    out = o==0
    out = out.float()
    return out*tau    

def dgdu(u):
    hu = abs( u - Vth ) < aa
    hu = hu.float() / (2 * aa)
    return hu

def main():
    torch.set_printoptions(precision=6)
    data = torch.randint(0, 2, (4,1,30,step),dtype=torch.float)
    torch.save(data,'data.pt')
    target = torch.tensor([[1,0],
                           [0,1],
                           [1,0],
                           [0,1]] , dtype=torch.float)
    target = target.reshape((4,1,2))
    data = torch.load('data.pt')
    device = torch.device('cpu')
    # SNN = SpikeNN()
    # model = SNN.to(device)
    # torch.save(model,'model_bp.pt')
    model = torch.load('model_bp.pt')
    optimizer = optim.Adam(params=[model.fc3.weight], lr= 1)
    params = model.fc3.weight
    # print('lr:',optimizer.state_dict())
    pass_ed = 0
    for i in range(100):
        for j in range(4):
            data_in = data[j,:,:,:]
            target_in = target[j,:,:]
            s1 = model.fc3.weight.data.clone().T
            # s = model.fc3.weight.data.detach().T
            # print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
            # print('s1')
            # print(s1)
            # optimizer.zero_grad()
            if params.grad is not None:
                params.grad.zero_()
            output, fc3_u_list  ,fc3_out_list = model(data_in )
            loss = F.mse_loss(output, target_in)
            # make_dot(loss.mean())
            loss.backward()
            with torch.no_grad():
                # print('params.grad')
                # print(params.grad)
                params -= 1 * params.grad
            # print('s2')
            # print(model.fc3.weight.data.detach().T)
            s2 = model.fc3.weight.data.clone().T
            # print('s2-s1')
            # print(s2-s1)
            # print('loss')
            # print(loss)
            # print('output')
            # print(output)
            # print('label')
            # print(target_in)
            # print('data_in')
            # print(data_in)
            # print('fc3_u_list')
            # print(fc3_u_list)
            # print('fc3_out_list')
            # print(fc3_out_list)
            # print('none')
            delta_error = []
            delta_u =[]
            for k in range(step):
                if k==0:
                    er = -1/step*(target_in-output)
                    u = fc3_u_list[step-1-k]
                    du = er*dgdu(u)
                    delta_w = data_in[:,:,step-1-k].T*du
                else:
                    u = fc3_u_list[step-1-k]
                    o = fc3_out_list[step-1-k]
                    er = delta_u[k-1]*u*(-tau) + delta_error[0]
                    du = er*dgdu(u) + delta_u[k-1]*f(o)
                    delta_w += data_in[:,:,step-1-k].T*du
                delta_error.append(er)
                delta_u.append(du)
            # print(delta_w)
            # print(delta_error)
            # print(delta_u)
            if torch.abs((delta_w+s2-s1)).max()>0.000001 :
                # print('pass')
                pass_ed+=1
            # else :
            #     print('nopasssssssss')
            # print(delta_w+s2-s1)
    print('PASS ? ' , pass_ed==0)

if __name__ == '__main__':
    main()
