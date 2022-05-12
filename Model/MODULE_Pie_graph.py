import matplotlib.pyplot as plt
import os

def Draw_pie_graphs(DATA, LABELS, TITLES, EXPLODES, START_ANGLE, LEGENDS, N_ROW = 1, AUTOPCT = "%.2f%%", FONTSIZE=10, SHADOW = True, FIGSIZE = (10,10),SHOW = False, DIR = "./Results/Pie_graphs", FILENAME = "pie_graph.png"):
    chart_num = len(DATA)
    tt_num = len(LABELS)
    plt.figure(figsize=FIGSIZE)
    for i in range(chart_num):
        plt.subplot(N_ROW,int(chart_num/N_ROW),i+1)
        if LEGENDS[i]:
            plt.pie(DATA[i], explode=EXPLODES, labels=LABELS, textprops={'fontsize':FONTSIZE}, shadow= SHADOW, startangle=START_ANGLE[i],radius=1)
            lbs = list()
            for j in range(tt_num):
                lbs.append(LABELS[j]+'('+str(round(DATA[i][j]/sum(DATA[i])*100,2))+'%)')
                plt.legend(lbs,loc='best')
        else:
            plt.pie(DATA[i], explode=EXPLODES, labels=LABELS, autopct=AUTOPCT, textprops={'fontsize':FONTSIZE}, shadow= SHADOW, startangle=START_ANGLE[i],radius=1)
        plt.title(TITLES[i])
    if SHOW:
        plt.show()
    else:
        resultdir = os.path.join(DIR,FILENAME)
        print(f'Pie graph saved to {resultdir}.\n')
        plt.savefig(resultdir,bbox_inches='tight')


if __name__ == '__main__':
    #an example
    data = [[36.00, 15.94, 15.71, 15.91, 15.65],
            [76.83, 5.55, 5.84, 5.8, 5.99],
            [88.86, 2.56, 2.71, 3.14, 2.73],
            [90.89, 2.25, 2.09, 2.49, 2.28]]
    labels =[ 'Correct Prediction', 'lvl1_err','lvl2_err', 'lvl3_err','lvl4_err']
    tts = ['1 model','2 model','3 model','4 model','5 model']
    explode = [0, 0.1, 0.2, 0.2, 0.1]
    starta = [0,0,0,0,0]
    leg = [False,False, True, True]
    
    Draw_pie_graphs(data, labels, tts, explode, starta, leg, 1, FIGSIZE=(20,20), SHOW=True)
