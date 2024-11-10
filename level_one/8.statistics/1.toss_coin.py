import random
import matplotlib.pyplot as plt

def toss_coin():
    return 'H' if random.random() < 0.5 else 'T'

N = 10
results = [toss_coin() for _ in range(N)]

print(results)

heads = results.count('H')
tails = results.count('T')

pmf = [heads/N, tails/N]

labels = ["Heads", "Tails"]
plt.bar(labels, pmf, color=['r', 'b'])
plt.title("Probability Mass Function of Coin Toss")
plt.ylabel("Probability")
plt.show()