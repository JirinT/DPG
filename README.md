# DPG
Unlike a stochastic policy, deterministic policy maps state S directly to action A: $u_{theta}: S->A$.

The policy is deterministic, but the environment can still be stochastic.

If we consider ON-Policy deterministic policy learning, then there is no inherent exploration. We then need to either assume that the environment is noisy, which would bring sort of exploration, or add artificial noise to our actions. For example gaussian distribution noise, which has zero mean. But better shaped noise is Ornstein-Uhlenbeck process, which is basically low pass filtering the gausian noise, so the mean does not have to be zero in a shorter period of time. This provides us more exploration with actions being positive or negative for a longer period of time. The OU process is:

![Equation](https://latex.codecogs.com/svg.image?&space;a_{k&plus;1}=\lambda&space;a_k&plus;\sigma\varepsilon&space;)

Since the action space is continuous, the policy is moved in the direction of Q function's gradient. Instead of maximizing Q by taking argmax(Q) action like in Q-learning, in DPG the policy will converge to maximal Q value. This Q function's gradient can be expressed as an expectation over behavioral state distribution: 

![Equation](https://latex.codecogs.com/svg.image?\mathbb{E}_{s%20\sim%20\rho^{\mu^k}}%20\left[%20\nabla_{\theta}%20Q^{\mu^k}%20\left(%20s,%20\mu_\theta%20(s)%20\right)%20\right])

Then by applying the chain rule, we see the gradient of Q is wrt actions and gradient of policy u is wrt theta:

![Equation](https://latex.codecogs.com/svg.image?\mathbb{E}_{s\sim%20\rho^{\mu%20k}}%20\left[%20\nabla_{\theta}%20\mu_{\theta}(s)%20\cdot%20\nabla_{a}%20Q^{\mu^{k}}(s,%20a)%20\bigg|_{a=\mu_{\theta}(s)}%20\right])

When expanded the final objective function is:

![Equation](https://latex.codecogs.com/svg.image?\nabla%20_%20{%20\theta%20}%20J%20(%20\mu%20_%20{%20\theta%20}%20)%20=%20\int%20_%20{%20S%20}%20\rho%20^%20{%20\mu%20}%20(%20s%20)%20\nabla%20_%20{%20\theta%20}%20\mu%20_%20{%20\theta%20}%20(%20s%20)%20\,%20\nabla%20_%20{%20a%20}%20Q%20^%20{%20\mu%20}%20(%20s%20,%20a%20)%20|%20_%20{%20a%20=%20\mu%20_%20{%20\theta%20}%20(%20s%20)%20}%20\,%20\mathrm%20{%20d%20}%20s)

We want to change the policy parameters in a direction of actions which will increase q values and therefore increase decision making performance.

The DPG implemented is COPDAC-Q algorithm and is described originally here: http://proceedings.mlr.press/v32/silver14.pdf and here on medium platform: https://medium.com/geekculture/introduction-to-deterministic-policy-gradient-dpg-e7229d5248e2
It uses linear approximations for the policy, Q function, and value function.