# DPG
Unlike a stochastic policy, deterministic policy maps state S directly to action A: $ny_theta: S->A$.

Since the action space is continuous, the policy is moved in the direction of Q function's gradient. Instead of maximizing Q by taking argmax(Q) action like in Q-learning, in DPG the policy will converge to maximal Q value. This Q function's gradient can be expressed as an expectation over behavioral state distribution: ![Equation](https://latex.codecogs.com/svg.image?\mathbb{E}_{s%20\sim%20\rho^{\mu^k}}%20\left[%20\nabla_{\theta}%20Q^{\mu^k}%20\left(%20s,%20\mu_\theta%20(s)%20\right)%20\right])
Then by applying the chain rule, we see the gradient of Q is wrt actions and gradient of policy u is wrt theta.