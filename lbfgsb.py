import torch
from functools import reduce
from torch.optim.optimizer import Optimizer

import math

be_verbose=False

class LBFGSB(Optimizer):
    """Implements L-BFGS-B algorithm.
     Primary reference:
      1) MATLAB code https://github.com/bgranzow/L-BFGS-B by Brian Granzow
     Theory based on:
      1) A Limited Memory Algorithm for Bound Constrained Optimization, Byrd et al. 1995
      2) Numerical Optimization, Nocedal and Wright, 2006

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. note::
        This is still WIP, the saving/restoring of state dict is not fully implemented.

    Arguments:
        lower_bound (shape equal to parameter vector): parameters > lower_bound
        upper_bound (shape equal to parameter vector): parameters < upper_bound
        max_iter (int): maximal number of iterations per optimization step
            (default: 10)
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-20).
        history_size (int): update history size (default: 7).
        batch_mode: True for stochastic version (default: False)
        cost_use_gradient: set this to True when the cost function also needs the gradient, for example in TV (total variation) regularization. (default: False)

    Example:
    ------
    >>> x=torch.rand(2,requires_grad=True,dtype=torch.float64,device=mydevice)
    >>> x_l=torch.ones(2,device=mydevice)*(-1.0)
    >>> x_u=torch.ones(2,device=mydevice)
    >>> optimizer=LBFGSB([x],lower_bound=x_l, upper_bound=x_u, history_size=7, max_iter=4, batch_mode=True)
    >>> def cost_function():
    >>>   f=torch.pow(1.0-x[0],2.0)+100.0*torch.pow(x[1]-x[0]*x[0],2.0)
    >>>   return f
    >>> for ci in range(10):
    >>>   def closure():
    >>>     if torch.is_grad_enabled():
    >>>       optimizer.zero_grad()
    >>>     loss=cost_function()
    >>>     if loss.requires_grad:
    >>>       loss.backward()
    >>>     return loss
    >>>   
    >>>   optimizer.step(closure)
    ------
    """

    def __init__(self, params, lower_bound, upper_bound, max_iter=10,
                 tolerance_grad=1e-5, tolerance_change=1e-20, history_size=7,
                 batch_mode=False, cost_use_gradient=False):
        defaults = dict(max_iter=max_iter,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size,
                        batch_mode=batch_mode,
                        cost_use_gradient=cost_use_gradient)
        super(LBFGSB, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSB doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._device = self._params[0].device
        self._dtype= self._params[0].dtype
        self._l=lower_bound.clone(memory_format=torch.contiguous_format).to(self._device)
        self._u=upper_bound.clone(memory_format=torch.contiguous_format).to(self._device)
        self._m=history_size
        self._n=self._numel()
        # local storage as matrices (instead of curvature pairs)
        self._W=torch.zeros(self._n,self._m*2,dtype=self._dtype).to(self._device)
        self._Y=torch.zeros(self._n,self._m,dtype=self._dtype).to(self._device)
        self._S=torch.zeros(self._n,self._m,dtype=self._dtype).to(self._device)
        self._M=torch.zeros(self._m*2,self._m*2,dtype=self._dtype).to(self._device)

        self._fit_to_constraints()

        self._eps=tolerance_change
        self._realmax=1e20
        self._theta=1

        # batch mode
        self.running_avg=None
        self.running_avg_sq=None
        self.alphabar=1.0

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().contiguous().view(-1)
            else:
                view = p.grad.data.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(update[offset:offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    #copy the parameter values out, create a list of vectors
    def _copy_params_out(self):
        return [p.flatten().clone(memory_format=torch.contiguous_format) for p in self._params]

    #copy the parameter values back, dividing the list appropriately
    def _copy_params_in(self,new_params):
        for p, pdata in zip(self._params, new_params):
            p.data.copy_(pdata.view_as(p).data)

    # restrict parameters to constraints 
    def _fit_to_constraints(self):
        params=[]
        for p in self._params:
            # make a vector
            p = p.flatten()
            params.append(p)
        x=torch.cat(params,0)
        for i in range(x.numel()):
          if (x[i]<self._l[i]):
              x[i]=self._l[i]
          elif (x[i]>self._u[i]):
              x[i]=self._u[i]
        offset = 0
        for p in self._params:
          numel = p.numel()
          p.data.copy_(x[offset:offset + numel].view_as(p.data))
          offset += numel
        assert offset == self._numel()

    def _get_optimality(self,g):
        # get the inf-norm of the projected gradient
        # pp. 17, (6.1)
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        x=torch.cat(self._copy_params_out(),0).detach()
        projected_g=x-g
        for i in range(x.numel()):
            if projected_g[i]<self._l[i]:
                projected_g[i]=self._l[i]
            elif projected_g[i]>self._u[i]:
                projected_g[i]=self._u[i]
        projected_g=projected_g-x
        return max(abs(projected_g))

    def _get_breakpoints(self,x,g):
        # compute breakpoints for Cauchy point
        # pp 5-6, (4.1), (4.2), pp. 8, CP initialize \mathcal{F}
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # out:
        # t: nx1 breakpoint vector
        # d: nx1 search direction vector
        # F: nx1 indices that sort t from low to high
        t=torch.zeros(self._n,1,dtype=self._dtype,device=self._device)
        d=-g
        for i in range(self._n):
            if (g[i]<0.0):
                t[i]=(x[i]-self._u[i])/g[i]
            elif (g[i]>0.0):
                t[i]=(x[i]-self._l[i])/g[i]
            else:
                t[i]=self._realmax

            if (t[i]<self._eps):
                d[i]=0.0

        F=torch.argsort(t.squeeze())

        return t,d.unsqueeze(-1),F

    def _get_cauchy_point(self,g):
        # Generalized Cauchy point
        # pp. 8-9, algorithm CP
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # theta: >0, scaling
        # W: nx2m 
        # M: 2mx2m
        # out:
        # xc: nx1 the generalized Cauchy point
        # c: 2mx1 initialization vector for subspace minimization

        x=torch.cat(self._copy_params_out(),0).detach()
        tt,d,F=self._get_breakpoints(x,g)
        xc=x.clone()
        c=torch.zeros(2*self._m,1,dtype=self._dtype,device=self._device)
        p=torch.mm(self._W.transpose(0,1),d)
        fp=-torch.mm(d.transpose(0,1),d)
        fpp=-self._theta*fp-torch.mm(p.transpose(0,1),torch.mm(self._M,p))
        fp=fp.squeeze()
        fpp=fpp.squeeze()
        fpp0=-self._theta*fp
        if (fpp != 0.0):
          dt_min=-fp/fpp
        else:
          dt_min=-fp/self._eps
        t_old=0
        # find lowest index i where F[i] is positive (minimum t)
        for j in range(self._n):
            i=j
            if F[i]>=0.0:
                break
        b=F[i]
        t=tt[b]
        dt=t-t_old

        while (i<self._n) and (dt_min>dt):
            if d[b]>0.0:
                xc[b]=self._u[b]
            elif d[b]<0.0:
                xc[b]=self._l[b]

            zb=xc[b]-x[b]
            c=c+dt*p
            gb=g[b]
            Wbt=self._W[b,:]
            Wbt=Wbt.unsqueeze(-1).transpose(0,1)
            fp=fp+dt*fpp+gb*gb+self._theta*gb*zb-gb*torch.mm(Wbt,torch.mm(self._M,c))
            fpp=fpp-self._theta*gb*gb-2.0*gb*torch.mm(Wbt,torch.mm(self._M,p))-gb*gb*torch.mm(Wbt,torch.mm(self._M,Wbt.transpose(0,1)))
            fp=fp.squeeze()
            fpp=fpp.squeeze()
            fpp=max(self._eps*fpp0,fpp)
            p=p+gb*Wbt.transpose(0,1)
            d[b]=0.0
            if (fpp != 0.0):
              dt_min=-fp/fpp
            else:
              dt_min=-fp/self._eps
            t_old=t
            i=i+1
            if i<self._n:
                b=F[i]
                t=tt[b]
                dt=t-t_old
          
        dt_min=max(dt_min,0.0)
        t_old=t_old+dt_min
        for j in range(i,self._n):
            idx=F[j]
            xc[idx]=x[idx]+t_old*d[idx]

        c = c + dt_min*p

        return xc,c

    def _subspace_min(self,g,xc,c):
        # subspace minimization for the quadratic model over free variables
        # direct primal method, pp 12
        # x: nx1 parameters
        # g: nx1 gradient
        # l: nx1 lower bound
        # u: nx1 upper bound
        # xc: nx1 generalized Cauchy point
        # c: 2mx 1 minimization initialization vector
        # theta: >0, scaling
        # W: nx2m 
        # M: 2mx2m
        # out:
        # xbar: nx1 minimizer 
        # line_search_flag: bool

        line_search_flag=True
        free_vars_index=list()
        Z=list()
        for i in range(self._n):
            if (xc[i] != self._u[i]) and (xc[i] != self._l[i]):
                free_vars_index.append(i)
                unit=torch.zeros(self._n,1,dtype=self._dtype,device=self._device)
                unit[i,0]=1
                Z.append(unit)
        n_free_vars=len(free_vars_index)
        if n_free_vars==0:
            xbar=xc.clone()
            line_search_flag=False
            return xbar,line_search_flag

        # Z: n x n_free_vars
        Za=torch.cat(Z,1)
        
        WtZ=torch.mm(self._W.transpose(0,1),Za)

        x=torch.cat(self._copy_params_out(),0).detach()
        rr=g+self._theta*(xc-x) - torch.mm(self._W,torch.mm(self._M,c)).squeeze()
        r=torch.zeros(n_free_vars,1,dtype=self._dtype,device=self._device)
        for i in range(n_free_vars):
            r[i]=rr[free_vars_index[i]]

        invtheta=1.0/self._theta
        v=torch.mm(self._M,torch.mm(WtZ,r))
        N=invtheta*torch.mm(WtZ,WtZ.transpose(0,1))
        N=torch.eye(2*self._m).to(self._device)-torch.mm(self._M,N)
        v,_,_,_=torch.linalg.lstsq(N,v,rcond=-1)
        du=-invtheta*r-invtheta*invtheta*torch.mm(WtZ.transpose(0,1),v)

        alpha_star=self._find_alpha(xc,du,free_vars_index)
        d_star=alpha_star*du
        xbar=xc.clone()
        for i in range(n_free_vars):
            idx=free_vars_index[i]
            xbar[idx]=xbar[idx]+d_star[i]

        return xbar,line_search_flag

    def _find_alpha(self, xc, du, free_vars_index):
        # pp. 11, (5.8)
        # l: nx1 lower bound
        # u: nx1 upper bound
        # xc: nx1 generalized Cauchy point
        # du: n_free_varsx1
        # free_vars_index:  n_free_varsx1 indices of free variables
        # out:
        # alpha_star: positive scaling parameter

        n_free_vars=len(free_vars_index)
        alpha_star=1.0
        for i in range(n_free_vars):
            idx=free_vars_index[i]
            if du[i]>0.0:
                alpha_star=min(alpha_star,(self._u[idx]-xc[idx])/du[i])
            elif du[i]<0.0:
                alpha_star=min(alpha_star,(self._l[idx]-xc[idx])/du[i])

        return alpha_star


    def _linesearch_backtrack(self, closure, f_old, gk, pk, alphabar):
        """Line search (backtracking)

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            f_old: original cost
            gk: gradient vector
            pk: step direction vector
            alphabar: max step size
        """
        c1=1e-4
        citer=35
        alphak=alphabar

        x0list=self._copy_params_out()
        xk=[x.detach().clone() for x in x0list]
        self._add_grad(alphak,pk)
        f_new=float(closure())
        s=gk
        prodterm=c1*s.dot(pk)
        ci=0
        while (ci<citer and (math.isnan(f_new) or f_new>f_old+alphak*prodterm)):
            alphak=0.5*alphak
            self._copy_params_in(xk)
            self._add_grad(alphak,pk)
            f_new=float(closure())
            ci=ci+1

        self._copy_params_in(xk)
        return alphak


    def _strong_wolfe(self, closure, f0, g0, p):
        # line search to satisfy strong Wolfe conditions
        # Alg 3.5, pp. 60, Numerical optimization Nocedal & Wright
        # cost: cost function R^n -> 1
        # gradient: gradient function R^n -> R^n
        # x0: nx1 initial parameters
        # f0: 1 intial cost
        # g0: nx1 initial gradient
        # p: nx1 intial search direction
        # out:
        # alpha: step length

        c1=1e-4
        c2=0.9
        alpha_max=2.5
        alpha_im1=0
        alpha_i=1
        f_im1=f0
        dphi0=torch.dot(g0,p)

        # make a copy of original params
        x0list=self._copy_params_out()
        x0=[x.detach().clone() for x in x0list]

        i=0
        max_iters=20
        while 1:
            # x=x0+alpha_i*p
            self._copy_params_in(x0)
            self._add_grad(alpha_i,p)
            f_i=float(closure())
            g_i=self._gather_flat_grad()
            if (f_i>f0+c1*dphi0) or ((i>0) and (f_i>f_im1)):
                alpha=self._alpha_zoom(closure,x0,f0,g0,p,alpha_im1,alpha_i)
                break
            dphi=torch.dot(g_i,p)
            if (abs(dphi)<=-c2*dphi0):
                alpha=alpha_i
                break
            if (dphi>=0.0):
                alpha=self._alpha_zoom(closure,x0,f0,g0,p,alpha_i,alpha_im1)
                break
            alpha_im1=alpha_i
            f_im1=f_i
            alpha_i=alpha_i+0.8*(alpha_max-alpha_i)
            if (i>max_iters):
                alpha=alpha_i
                break
            i=i+1

        # restore original params
        self._copy_params_in(x0)
        return alpha


    def _alpha_zoom(self, closure, x0, f0, g0, p, alpha_lo, alpha_hi):
        # Alg 3.6, pp. 61, Numerical optimization Nocedal & Wright
        # cost: cost function R^n -> 1
        # gradient: gradient function R^n -> R^n
        # x0: list() initial parameters
        # f0: 1 intial cost
        # g0: nx1 initial gradient
        # p: nx1 intial search direction
        # alpha_lo: low limit for alpha
        # alpha_hi: high limit for alpha
        # out:
        # alpha: zoomed step length
        c1=1e-4
        c2=0.9
        i=0
        max_iters=20
        dphi0=torch.dot(g0,p)
        while 1:
            alpha_i=0.5*(alpha_lo+alpha_hi)
            alpha=alpha_i
            # x=x0+alpha_i*p
            self._copy_params_in(x0)
            self._add_grad(alpha_i,p)
            f_i=float(closure())
            g_i=self._gather_flat_grad()
            # x_lo=x0+alpha_lo*p
            self._copy_params_in(x0)
            self._add_grad(alpha_lo,p)
            f_lo=float(closure())
            if ((f_i>f0+c1*alpha_i*dphi0) or (f_i>=f_lo)):
                alpha_hi=alpha_i
            else:
                dphi=torch.dot(g_i,p)
                if ((abs(dphi)<=-c2*dphi0)):
                    alpha=alpha_i
                    break
                if (dphi*(alpha_hi-alpha_lo)>=0.0):
                    alpha_hi=alpha_lo
                alpha_lo=alpha_i
            i=i+1
            if (i>max_iters):
                alpha=alpha_i
                break

        return alpha




    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group['max_iter']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        batch_mode = group['batch_mode']
        cost_use_gradient = group['cost_use_gradient']


        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)


        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        f= float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        g=self._gather_flat_grad()
        abs_grad_sum = g.abs().sum()

        if torch.isnan(abs_grad_sum) or abs_grad_sum <= tolerance_grad:
            return orig_loss

        n_iter=0

        if batch_mode and state['n_iter']==0:
            self.running_avg=torch.zeros_like(g.data)
            self.running_avg_sq=torch.zeros_like(g.data)

        while (self._get_optimality(g)>tolerance_change) and  n_iter<max_iter:
            x_old=torch.cat(self._copy_params_out(),0).detach()
            g_old=g.clone()
            xc,c=self._get_cauchy_point(g)
            xbar,line_search_flag=self._subspace_min(g,xc,c)
            alpha=1.0
            p=xbar-x_old
            if (line_search_flag):
                if not batch_mode:
                  alpha=self._strong_wolfe(closure,f,g,p)
                else:
                  if not cost_use_gradient:
                        torch.set_grad_enabled(False)
                  alpha=self._linesearch_backtrack(closure,f,g,p,self.alphabar)
                  if not cost_use_gradient:
                        torch.set_grad_enabled(True)

            self._add_grad(alpha,p)

            f=float(closure())
            g=self._gather_flat_grad()
            y=g-g_old
            x=torch.cat(self._copy_params_out(),0).detach()
            s=x-x_old
            curv=abs(torch.dot(s,y))
            n_iter +=1
            state['n_iter'] +=1


            batch_changed=batch_mode and (n_iter==1 and state['n_iter']>1)
            if batch_changed:
                tmp_grad_1=g_old.clone(memory_format=torch.contiguous_format)
                tmp_grad_1.add_(self.running_avg,alpha=-1.0)
                self.running_avg.add_(tmp_grad_1,alpha=1.0/state['n_iter'])
                tmp_grad_2=g_old.clone(memory_format=torch.contiguous_format)
                tmp_grad_2.add_(self.running_avg,alpha=-1.0)
                self.running_avg_sq.addcmul_(tmp_grad_2,tmp_grad_1,value=1)
                self.alphabar=1.0/(1.0+self.running_avg_sq.sum()/((state['n_iter']-1)*g_old.norm().item()))


            if (curv<self._eps):
                print('Warning: negative curvature detected, skipping update')
                n_iter+=1
                continue
            # in batch mode, do not update Y and S if the batch has changed
            if not batch_changed:
                if (n_iter<self._m):
                    self._Y[:,n_iter]=y.squeeze()
                    self._S[:,n_iter]=s.squeeze()
                else:
                    self._Y[:,0:self._m-1]=self._Y[:,1:self._m]
                    self._S[:,0:self._m-1]=self._S[:,1:self._m]
                    self._Y[:,-1]=y.squeeze()
                    self._S[:,-1]=s.squeeze()

                self._theta=torch.dot(y,y)/torch.dot(y,s)
                self._W[:,0:self._m]=self._Y
                self._W[:,self._m:2*self._m]=self._theta*self._S
                A=torch.mm(self._S.transpose(0,1),self._Y)
                L=torch.tril(A,-1)
                D=-1.0*torch.diag(torch.diag(A))
                MM=torch.zeros(2*self._m,2*self._m,dtype=self._dtype,device=self._device)
                MM[0:self._m,0:self._m]=D
                MM[0:self._m,self._m:2*self._m]=L.transpose(0,1)
                MM[self._m:2*self._m,0:self._m]=L
                MM[self._m:2*self._m,self._m:2*self._m]=self._theta*torch.mm(self._S.transpose(0,1),self._S)
                self._M=torch.linalg.pinv(MM) 



       

        if be_verbose and (n_iter==max_iter):
            print('Reached maximum number of iterations,  stopping')

        if be_verbose and (self._get_optimality(g)<self._eps):
            print('Reached required convergence tolerance, stopping')

        return f
