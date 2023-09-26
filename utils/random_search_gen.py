import random

ke_range = range(0, 501)
buffer_range = range(0, 2501)
alpha_range = range(0, 1000)
beta_range = range(0, 100)


def main():
    ke = random.choice(ke_range)
    buffer = random.choice(buffer_range)
    alpha = random.choice(alpha_range)
    beta = random.choice(beta_range)
    mole_coll = random.random()
    ke_loss_rate = random.random()
    print(f'{ke=}')
    print(f'{buffer=}')
    print(f'{alpha=}')
    print(f'{beta=}')
    print(f'{mole_coll=}')
    print(f'{ke_loss_rate=}')


if __name__ == "__main__":
    main()
