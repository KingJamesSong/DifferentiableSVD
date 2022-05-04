'''
@author: Yue Song
'''
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as Fa

def ordinary_gradients(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dim = s.size(1)
    p = 1 / (s.unsqueeze(-1) - s.unsqueeze(-2))
    p[:,torch.arange(0,dim),torch.arange(0,dim)]=0
    return p

class SEB(nn.Module):
     """Scaling Eigen Branch that amplifies the relative improtance of small eigenvalues
        see T-PAMI paper: On the Eigenvalues of Global Covariance Pooling for Fine-grained Visual Recognition
     Args:
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(self, is_vec=True, input_dim=2048, dimension_reduction=None):

         super(SEB, self).__init__()
         self.is_vec = is_vec
         self.dr = dimension_reduction
         self.softmax_ = nn.Softmax(dim=1)
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
     def _pow(self, x1,x2):
         return Power.apply(x1,x2)
     def _eig(self, x):
         return Eigen_decomposition.apply(x)
     def _triu(self,x1):
         return Triuvec.apply(x1)
     def _expinv(self,x1,x2):
         return ExponentialInverse.apply(x1,x2)

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._cov_pool(x)
         eig_vec, eig_diag = self._eig(x)
         inverse_x= self._expinv(eig_vec,eig_diag) #exponential inverse
         out = self._pow(eig_vec,eig_diag) #matrix square root
         attention = out.bmm(inverse_x.transpose(1,2)) #QS^{T}
         norm = Fa.normalize(attention.view(out.size(0),-1),dim=1)
         attention = out * (attention / norm.view(out.size())) #multiply ||QS^{T}||_{F}
         out =  self._triu(attention + out)
         return out


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        input = input.double()  # Change the spectral layer into double precision to assure effective numercial representation
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w).float()  # Change back to float
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
         eig_diag[eig_diag <= torch.finfo(dtype).eps] = torch.finfo(dtype).eps #Zero-out eigenvalues smaller than eps
         eig_diag=eig_diag.diag_embed().type(dtype)
         ctx.save_for_backward(eig_vec,eig_diag)
         return eig_vec,eig_diag
     @staticmethod
     def backward(ctx, grad_output1,grad_output2):
         eig_vec,eig_diag = ctx.saved_tensors
         eig_vec_deri,eig_diag_deri=grad_output1,grad_output2
         k = ordinary_gradients(eig_diag)
         #Gradient Overflow Check;
         k[k==float('inf')]=k[k!=float('inf')].max()
         k[k==float('-inf')]=k[k!=float('-inf')].min()
         k[k!=k]=k.max()
         grad_input=(k.transpose(1,2)*(eig_vec.transpose(1,2).bmm(eig_vec_deri)))+torch.diag_embed(torch.diagonal(eig_diag_deri,dim1=1,dim2=2))
         # Gradient Overflow Check;
         grad_input[grad_input==float('inf')]=grad_input[grad_input!=float('inf')].max()
         grad_input[grad_input==float('-inf')]=grad_input[grad_input!=float('-inf')].min()
         grad_input=eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1,2))
         # Gradient Overflow Check;
         grad_input[grad_input==float('inf')]=grad_input[grad_input!=float('inf')].max()
         grad_input[grad_input==float('-inf')]=grad_input[grad_input!=float('-inf')].min()
         return grad_input

# Exponential Inverse to balance eigenvalue distribution
class ExponentialInverse(Function):
    @staticmethod
    def forward(ctx, input1,input2):
        eig_vec, eig_diag = input1, input2
        dia_eig_diag = torch.diagonal(eig_diag, dim1=1, dim2=2)
        inverse_eig = torch.exp(-dia_eig_diag)
        inverse_eig = torch.diag_embed(inverse_eig)
        inverse_x = eig_vec.bmm(inverse_eig).bmm(eig_vec.transpose(1, 2))
        ctx.save_for_backward(eig_vec, eig_diag)
        return inverse_x
    @staticmethod
    def backward(ctx, grad_output):
        eig_vec, eig_diag = ctx.saved_tensors
        dia_eig_diag = torch.diagonal(eig_diag, dim1=1, dim2=2)
        inverse_eig = torch.exp(-dia_eig_diag)
        inverse_eig = torch.diag_embed(inverse_eig)
        batch_size = eig_diag.data.shape[0]
        dim = eig_diag.data.shape[1]
        dtype = eig_diag.dtype
        grad_output_all = grad_output
        grad_output_all = grad_output_all.reshape(batch_size, dim, dim).type(dtype)
        grad_input1 = (grad_output_all + grad_output_all.transpose(1, 2)).bmm(eig_vec).bmm(inverse_eig)
        grad_input2 = (-inverse_eig).bmm(eig_vec.transpose(1, 2)).bmm(grad_output_all).bmm(eig_vec)
        grad_input2 = torch.diag_embed(torch.diagonal(grad_input2, dim1=1, dim2=2))
        return grad_input1, grad_input2

# Matrix Square Root
class Power(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        eig_vec, eig_diag = input1, input2
        dtype = eig_vec.dtype
        power_eig_dia = eig_diag.sqrt().type(dtype)
        q = eig_vec.bmm(power_eig_dia).bmm(eig_vec.transpose(1, 2))
        ctx.save_for_backward(eig_vec, eig_diag)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        eig_vec, eig_diag = ctx.saved_tensors
        batch_size = eig_diag.data.shape[0]
        dim = eig_diag.data.shape[1]
        dtype = eig_diag.dtype
        grad_output_all = grad_output
        grad_output_all = grad_output_all.reshape(batch_size, dim, dim).type(dtype)
        grad_input1 = (grad_output_all + grad_output_all.transpose(1, 2)).bmm(eig_vec).bmm(eig_diag.pow(0.5))
        # No l2 or Frobenius norm
        power_eig = torch.diag_embed(torch.diagonal(eig_diag, dim1=1, dim2=2).pow(-0.5))
        grad_input2 = (power_eig).bmm(eig_vec.transpose(1, 2)).bmm(grad_output_all).bmm(eig_vec)
        grad_input2 = 0.5 * torch.diag_embed(torch.diagonal(grad_input2, dim1=1, dim2=2))
        return grad_input1, grad_input2

class Triuvec(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(dtype)
        y = x[:, index].float()
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(batchSize, dim * dim, device=x.device, requires_grad=False)
        grad_input[:, index] = grad_output
        grad_input = grad_input.reshape(batchSize, dim, dim).type(dtype)
        return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def EIGLayer(var):
    return Eigen_decomposition.apply(var)

def Powerlayer(var):
    return Power.apply(var)
