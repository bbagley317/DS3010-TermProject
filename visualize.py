from matplotlib import pyplot as plt, ticker

import numpy as np

import pandas as pd

 

def preprocess():  

    cand_comm_contrib = pd.DataFrame(pd.read_csv("./data/Contributions.csv"))

    candidates = pd.DataFrame(pd.read_csv("./data/Candidate master.csv"))

    candidates = candidates[["CAND_ID", "CAND_PTY_AFFILIATION", "CAND_ST"]]

 

    print(cand_comm_contrib.describe())

 

    data = pd.merge(cand_comm_contrib, candidates, on="CAND_ID")

    data.to_csv("data.csv",index=False)

   

def pie_chart():

    data = pd.DataFrame(pd.read_csv("./data.csv"))

 

    groups = data.groupby("CAND_PTY_AFFILIATION")["TRANSACTION_AMT"].sum().reset_index()

    groups = groups.sort_values(by="TRANSACTION_AMT", ascending=False).loc[groups["TRANSACTION_AMT"] > 10000]

 

    plt.figure(figsize=(10, 6))

 

    labels = groups["CAND_PTY_AFFILIATION"]

    sizes = groups["TRANSACTION_AMT"]

 

    explode = [0.1 if i == sizes.idxmax() else 0 for i in range(len(labels))]

 

    color_dict = {'REP': '#e06666ff', 'DEM': '#32adc3ff', 'DFL': 'green', 'IND': 'gray', 'NPP': 'purple', 'LIB': 'orange', 'GRE': 'violet', 'CRV': 'brown', 'Other': 'black'}

 

    colors = [color_dict.get(affiliation, 'orange') for affiliation in labels]

 

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)

    plt.title('Total Committee Contributions by Party Affiliation')

    plt.axis('equal')  

 

    plt.show()

 

def bar_graph():

    data = pd.DataFrame(pd.read_csv("./data.csv"))

 

    data['PARTY_CATEGORY'] = np.where(data['CAND_PTY_AFFILIATION'].isin(['REP', 'DEM']),

                                      data['CAND_PTY_AFFILIATION'], 'Other')

 

    party_totals = data.groupby('PARTY_CATEGORY')['TRANSACTION_AMT'].sum().reset_index()

    party_totals = party_totals.sort_values(by='TRANSACTION_AMT', ascending=False)

 

    plt.figure(figsize=(10, 6))

 

    bars = plt.bar(party_totals['PARTY_CATEGORY'], party_totals['TRANSACTION_AMT'], color=['#32adc3ff', '#e06666ff', 'gray'])

 

    for bar in bars:

        height = bar.get_height()

        plt.text(bar.get_x() + bar.get_width() / 2, height, f'${height:,.0f}',

                 ha='center', va='bottom', fontsize=8)

 

    plt.xlabel('Party Affiliation')

    plt.ylabel('Total Contributions (USD)')

    plt.title('Total Committee Contributions by Party Affiliation (Sorted by Contributions)')

 

    formatter = ticker.StrMethodFormatter('${x:,.0f}')

    plt.gca().yaxis.set_major_formatter(formatter)

 

    plt.xticks(rotation=45, ha='right')

 

    plt.tight_layout()

    plt.show()

 

# Note: Both the pie_chart function and the bar_graph function will not show up at the same time. To see the bar_graph you must close the pie_chart and it should open.

# If this does not work then comment out the graph you do not want to see. Sorry for the inconvience.

#preprocess()

pie_chart()

bar_graph()