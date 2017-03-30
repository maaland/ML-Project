import pandas as pd
import sys
import matplotlib.pyplot as plt

C = "cmybgrkw"

def main():
    print sys.argv
    if len(sys.argv) != 3:
        print "Usage: " + sys.argv[0] + " CSV TITLE"
        return
    
    data = pd.read_csv(sys.argv[1]).groupby('vectorizer')
    plt.figure()
    i = 0
    for name, df in data:
        print 80*"-"
        print name
        print df
        l1, = plt.plot(df['num_features'], df['train_accuracy'], '--',
                     linewidth=.5, label='', color=C[i])

        
        l2, = plt.plot(df['num_features'], df['test_accuracy'], '-',
                     linewidth=.5, label=name, color=C[i])

        #l1.set_antialiased(False)
        #l2.set_antialiased(False)
        
        i = i + 1
        #plt.plot(df['num_features'], df['train_accuracy'], label=name)
    plt.title(sys.argv[2])
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
