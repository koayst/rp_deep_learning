import matplotlib.pyplot as plt

def chart01(f, df, c):
    fig, axes = plt.subplots(nrows=len(f), figsize=(10, 10))
    for i in range(0, len(f.values)):
        axes[i].plot(df['date'], df[f.values[i]], color=c)
        axes[i].set_title(f.values[i])
        axes[i].grid(True)
    plt.subplots_adjust(hspace=0.8, )
    plt.show()
