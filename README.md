Good source: https://github.com/upb-lea/reinforcement_learning_course_materials/blob/master/exercises/templates/ex12/DDPG_and_PPO.ipynb
# DPG
Unlike a stochastic policy, deterministic policy maps state S directly to action A: $u_{theta}: S->A$.

The policy is deterministic, but the environment can still be stochastic.

If we consider ON-Policy deterministic policy learning, then there is no inherent exploration. We then need to either assume that the environment is noisy, which would bring sort of exploration, or add artificial noise to our actions. For example gaussian distribution noise, which has zero mean. But better shaped noise is Ornstein-Uhlenbeck process, which is basically low pass filtering the gausian noise, so the mean does not have to be zero in a shorter period of time. This provides us more exploration with actions being positive or negative for a longer period of time. The OU process is:

![Equation](https://latex.codecogs.com/svg.image?&space;a_{k&plus;1}=\lambda&space;a_k&plus;\sigma\varepsilon&space;)

Where $a$ is output noise (The action), $0<\lambda<1$ is a smoothing factor and $\sigma$ is variance scaling a standard Gaussian distribution.

Since the action space is continuous, the policy is moved in the direction of Q function's gradient. Instead of maximizing Q by taking argmax(Q) action like in Q-learning, in DPG the policy will converge to maximal Q value. This Q function's gradient can be expressed as an expectation over behavioral state distribution: 

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\LARGE%20\mathbb{E}_{s%20\sim%20\rho^{\mu^k}}%20\left[%20\nabla_{\theta}%20Q^{\mu^k}%20\left(%20s,%20\mu_\theta%20(s)%20\right)%20\right]" alt="Policy Gradient Equation">
</p>

Then by applying the chain rule, we see the gradient of Q is wrt actions and gradient of policy u is wrt theta:

![Equation](https://latex.codecogs.com/svg.image?\mathbb{E}_{s\sim%20\rho^{\mu%20k}}%20\left[%20\nabla_{\theta}%20\mu_{\theta}(s)%20\cdot%20\nabla_{a}%20Q^{\mu^{k}}(s,%20a)%20\bigg|_{a=\mu_{\theta}(s)}%20\right])

When expanded the final objective function is:

![Equation](https://latex.codecogs.com/svg.image?\nabla%20_%20{%20\theta%20}%20J%20(%20\mu%20_%20{%20\theta%20}%20)%20=%20\int%20_%20{%20S%20}%20\rho%20^%20{%20\mu%20}%20(%20s%20)%20\nabla%20_%20{%20\theta%20}%20\mu%20_%20{%20\theta%20}%20(%20s%20)%20\,%20\nabla%20_%20{%20a%20}%20Q%20^%20{%20\mu%20}%20(%20s%20,%20a%20)%20|%20_%20{%20a%20=%20\mu%20_%20{%20\theta%20}%20(%20s%20)%20}%20\,%20\mathrm%20{%20d%20}%20s)

We want to change the policy parameters in a direction of actions which will increase q values and therefore increase decision making performance.

The DPG implemented is COPDAC-Q algorithm and is described originally here: http://proceedings.mlr.press/v32/silver14.pdf and here on medium platform: https://medium.com/geekculture/introduction-to-deterministic-policy-gradient-dpg-e7229d5248e2
It uses linear approximations for the policy, Q function, and value function.

# DDPG
- It actually is DQN + DPG. Deep means that Q value is aproximated by neural network and policy as well.
## Usefull tweaks:
- **Experience replay buffer**
- **Target networks**, like in DQN where a target (delayed) network was used to compute the target Q-value which was used for updating Q-network parameters - it was updated to equall the training Q value every now and then. But now we need to introduce second target network for policy, because as can be seen in this equation (Loss used to update Q-network parameters w):

$$
\mathcal{L}(w) = \left[(r + \gamma q(x', \mu(x', \theta), w)) - q(x, u, w)\right]^{\frac{2}{D}}
$$

the target q value contains the policy, so to make it numerically stable, we need to put there delayed policy net.

- The target networks can be updated continuously (smoothly) by the equation:

$$
\theta^{-}←\left(1-\tau\right)\theta^{-}+\tau\theta
$$

which basically is low pass filter with filter constant Tau.

- **Mini-batch sampling** for updating the networks
- **Batch normalization**
- **Exploration rate** - Since the policy is deterministic, we require exploratory behavior policy. Standard approach is to add noise to the drawn actions from deterministic policy for example using Ornstein-Uhlenbeck process:

$$
u_k \sim b(u|x_k) = \mu(x_k, \theta_k) + \nu_k, \quad \nu_k = \lambda \nu_{k-1} + \sigma \epsilon_{k-1}
$$

- update of the policy values is done via the chain rule equation above. But in pytorch, we can just call q_value.backward() and it will compute both the policy gradient wrt its parameters and q-value gradient wrt action.

- the Q-network is updated (trained) using the actions from replay memory - the actions used in environment with added noise! But the policy network needs to be updated with deterministic actions corresponding to its current policy!!! So when updating policy, actions needs to be recomputed for current batch states and corresponding Q-values as well.
