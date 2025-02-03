import torch

class QValue:
    def __init__(self, parameters, policy, value_fcn):
        self.w = parameters
        self.policy = policy
        self.value_fcn = value_fcn
        self.fi = None
    
    def __call__(self, state, b_action):
        # Convert b_action to a PyTorch tensor if it's not already
        if not isinstance(b_action, torch.Tensor):
            b_action = torch.tensor(b_action, dtype=torch.float32)

        V = self.value_fcn(state)
        action = self.policy(state)
        action = action.requires_grad_()
        action_error = b_action - action # difference between the chosen action by behavioral policy and action which wopuld be chosen by our actuall policy

        # zero the gradients of theta in policy:
        if self.policy.theta.grad is not None:
            self.policy.theta.grad.zero_()

        action.backward()
        self.fi = self.policy.theta.grad*action_error
        A = torch.dot(self.fi, self.w) # advantage value - linear aproximation

        return A + V