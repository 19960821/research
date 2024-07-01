import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

file_path = "objective_time.csv"
df = pd.read_csv(file_path)

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(df["Episode"], df["DDPG_Obj"], label="DDPG")
plt.plot(df["Episode"], df["Exhaustive method_Obj"], label="Exhaustive method")
plt.xlabel("User number")
plt.ylabel("Objective")
plt.legend()
plt.show()
        

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(df["Episode"], df["DDPG_Time"], label="DDPG")
plt.plot(df["Episode"], df["Exhaustive method_Time"], label="Exhaustive method")
plt.xlabel("User number")
plt.ylabel("Execution time (s)")
plt.legend()
plt.show()

