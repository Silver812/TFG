from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Read the CSV data into a DataFrame
df = pd.read_csv("testing\\testing_data.csv")

# Set the figure size
plt.figure(figsize=(10, 10))

sns.set_style("darkgrid")

# Create a box plot with different colors for each box
sns.boxplot(x="algorithm", y="score", data=df, palette="Set1")

# Add jitter with the swarmplot function
sns.swarmplot(x="algorithm", y="score", data=df, color=".25", size=8)

# Add more y-ticks
plt.yticks(np.arange(df["score"].min(), df["score"].max() + 2, step=2))

# Change the x and y labels
plt.xlabel("Algoritmo", fontsize=14)
plt.ylabel("Puntuación", fontsize=14)

# # Add jitter with the swarmplot function
# sns.swarmplot(x="algorithm", y="errors", data=df, palette="Set1", size=8)

# # Add more y-ticks
# plt.yticks(np.arange(df["errors"].min(), df["errors"].max() + 1, step=1))

# # Change the x and y labels
# plt.xlabel("Algoritmo", fontsize=14)
# plt.ylabel("Número de errores", fontsize=14)

# Change the categories of the X axis
new_categories = ["PPO", "A2C", "PPO Extendido", "Jugador"]  # Example new categories
plt.xticks(ticks=np.arange(len(new_categories)), labels=new_categories)

# Show the plot
plt.show()
