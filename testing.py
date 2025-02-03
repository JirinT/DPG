from DDPG.replay_memory import ReplayMemory

a = (1,2,3)
b = (4,5,6)
c = (7,8,9)

mem = ReplayMemory(10)
mem.push(a)
mem.push(b)
mem.push(c)

batch = mem.sample(3)
print([item[1] for item in batch])