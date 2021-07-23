'''
@author: Yue Song
'''
import torch
import torch.nn as nn
from torch.autograd import Function

def truncated_gradients(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dim = s.size(1)
    #I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = 1 / (s.unsqueeze(-1) - s.unsqueeze(-2))
    p[:,torch.arange(0,dim),torch.arange(0,dim)]=0
    p[p > 1e10] = 1e10
    p[p < -1e10] = -1e10
    return p

class SVD_Trunc(nn.Module):
     """Forward Pass: SVD
        Backward Pass: SVD gradient + Truncation
     Args:
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(self, is_vec=True, input_dim=2048, dimension_reduction=None):

         super(SVD_Trunc, self).__init__()
         self.is_vec = is_vec
         self.dr = dimension_reduction
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
               nn.BatchNorm2d(self.dr),
               nn.ReLU(inplace=True)
             )
         output_dim = self.dr if self.dr else input_dim
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()

     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def _cov_pool(self, x):
         return Covpool.apply(x)
     def _eig(self, x):
         return Eigen_decomposition.apply(x)
     def _pow(self, x1,x2):
         return Power.apply(x1,x2)

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._cov_pool(x)
         eig_vec, eig_diag = self._eig(x)
         out = self._pow(eig_vec,eig_diag)
         return out


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         input=input.double() #Change the spectral layer into double precision to assure effective numercial representation
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x=input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         #Gradient Overflow Check;
         grad_input[grad_input==float('inf')]=grad_input[grad_input!=float('inf')].max()
         grad_input[grad_input==float('-inf')]=grad_input[grad_input!=float('-inf')].min()
         grad_input[grad_input!=grad_input]=0
         grad_input = grad_input.reshape(batchSize,dim,h,w).float() #Change back to float
         return grad_input

#SVD Step
class Eigen_decomposition(Function):
     @staticmethod
     def forward(ctx, input):
         p=input
         dtype=p.dtype
         p=p.cpu() #SVD is faster on CPU
         _,eig_diag,eig_vec=torch.svd(p, some=True, compute_uv=True)
         eig_diag=eig_diag.cuda()
         eig_vec=eig_vec.cuda()
         eig_diag[eig_diag <= torch.finfo(dtype).eps] = torch.finfo(dtype).eps
         eig_diag=eig_diag.diag_embed().type(dtype)
         ctx.save_for_backward(eig_vec,eig_diag)
         return eig_vec,eig_diag
     @staticmethod
     def backward(ctx, grad_output1,grad_output2):
         eig_vec,eig_diag = ctx.saved_tensors
         eig_vec_deri,eig_diag_deri=grad_output1,grad_output2
         k = truncated_gradients(eig_diag)
         grad_input=(k.transpose(1,2)*(eig_vec.transpose(1,2).bmm(eig_vec_deri)))+torch.diag_embed(torch.diagonal(eig_diag_deri,dim1=1,dim2=2))
         #Gradient Overflow Check
         grad_input[grad_input==float('inf')]=grad_input[grad_input!=float('inf')].max()
         grad_input[grad_input==float('-inf')]=grad_input[grad_input!=float('-inf')].min()
         grad_input=eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1,2))
         # Gradient Overflow Check
         grad_input[grad_input==float('inf')]=grad_input[grad_input!=float('inf')].max()
         grad_input[grad_input==float('-inf')]=grad_input[grad_input!=float('-inf')].min()
         return grad_input

#Matrix Square Root
class Power(Function):
     @staticmethod
     def forward(ctx, input1, input2):
         eig_vec, eig_diag=input1, input2
         batch_size=eig_diag.data.shape[0]
         dim=eig_diag.data.shape[1]
         dtype=eig_vec.dtype
         power_eig_dia=eig_diag.sqrt().type(dtype)
         q=eig_vec.bmm(power_eig_dia).bmm(eig_vec.transpose(1,2))
         q=q.reshape(batch_size,dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         q_triv=torch.zeros(batch_size,int(dim*(dim+1)/2),device = q.device)
         q_triv=q[:,index].float() #Change back to float precision
         ctx.save_for_backward(eig_vec,eig_diag,index)
         return q_triv
     @staticmethod
     def backward(ctx, grad_output):
         eig_vec,eig_diag,index = ctx.saved_tensors
         batch_size = eig_diag.data.shape[0]
         dim = eig_diag.data.shape[1]
         dtype = eig_diag.dtype
         grad_output_all=torch.zeros(batch_size,dim*dim,device = eig_diag.device,requires_grad=False)
         grad_output_all[:,index]=grad_output
         grad_output_all=grad_output_all.reshape(batch_size,dim,dim).type(dtype)
         grad_input1=(grad_output_all+grad_output_all.transpose(1,2)).bmm(eig_vec).bmm(eig_diag.pow(0.5))
         # No l2 or Frobenius norm
         power_eig=torch.diag_embed(torch.diagonal(eig_diag,dim1=1,dim2=2).pow(-0.5))
         grad_input2=(power_eig).bmm(eig_vec.transpose(1,2)).bmm(grad_output_all).bmm(eig_vec)
         grad_input2=0.5*torch.diag_embed(torch.diagonal(grad_input2,dim1=1,dim2=2))
         return grad_input1,grad_input2
        
def CovpoolLayer(var):
    return Covpool.apply(var)

def EIGLayer(var):
    return Eigen_decomposition.apply(var)

def Powerlayer(var):
    return Power.apply(var)