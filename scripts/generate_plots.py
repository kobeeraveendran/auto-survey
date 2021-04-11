import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("../plots/run_logs.csv", names = ["R-1 precision", "R-L precision", "R-1 recall", "R-L recall"])

    df.plot(
        title = "ROUGE scores w.r.t. Number of Topics", 
        xlabel = "Number of topics", 
        ylabel = "ROUGE score", 
        marker = 's', 
        linestyle = '--'
    )
    
    plt.savefig("../plots/rouge_scores.png")