import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV file
df = pd.read_csv('Iris.csv')  # 🔁 Replace with your actual filename

# Create output folders if not exist
os.makedirs('plots', exist_ok=True)
os.makedirs('summary', exist_ok=True)

# Display and save statistical summary
summary = df.describe()
print("Statistical Summary:\n", summary)

# Save summary as CSV and Excel
summary.to_csv('summary/statistical_summary.csv')
summary.to_excel('summary/statistical_summary.xlsx')

# Define columns to compare with Id
columns_to_compare = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Set seaborn style
sns.set(style="whitegrid")

# Generate and save plots
for col in columns_to_compare:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Id', y=col, data=df, marker='o')
    plt.title(f'Id vs {col}')
    plt.xlabel('Id')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'plots/id_vs_{col}.png')  # Save plot as image
    plt.close()

# Optional: Pairplot by Species
sns.pairplot(df, hue='Species', vars=columns_to_compare)
plt.suptitle("Feature Comparison by Species", y=1.02)
plt.savefig("plots/pairplot_by_species.png")
plt.close()

print("✅ Analysis complete: Summary saved in 'summary/', plots in 'plots/' folder.")
