from DDPG.ornstein_uhlbeck_noise import OU_noise

ou_noise = OU_noise(0.5, 0.3)

for _ in range(15):
    print(ou_noise.sample())