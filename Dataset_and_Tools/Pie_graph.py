from cProfile import label
from lib2to3.pgen2.token import PERCENT
import matplotlib.pyplot as plt

labels =[ 'Correct Prediction', 'lvl1_err','lvl2_err', 'lvl3_err','lvl4_err']
#VI
#size1 = [41.95, 12.8, 11.6, 16.4, 16.8]
#size2 = [74.83, 7.38, 5.03, 5.37, 7.38]
#size3 = [88.59, 1.68, 2.68, 3.36, 3.69]
#size4 = [90.6, 1.34, 2.35, 3.02, 2.68]

#VIII
#size1 = [35.69, 16.28, 16.14, 15.9, 15.99]
#size2 = [76.19, 5.86, 5.86, 6.24, 5.84]
#size3 = [87.94, 2.87, 2.97, 2.98, 3.24]
#size4 = [90.43, 2.44, 2.27, 2.32, 2.54]

#IX
size1 = [36.00, 15.94, 15.71, 15.91, 15.65]
size2 = [76.83, 5.55, 5.84, 5.8, 5.99]
size3 = [88.86, 2.56, 2.71, 3.14, 2.73]
size4 = [90.89, 2.25, 2.09, 2.49, 2.28]



explode = [0, 0.1, 0.2, 0.2, 0.1]

plt.subplot(1,4,1)
plt.pie(size1, explode=explode, labels=labels, autopct='%.2f%%', textprops={'fontsize':14}, shadow=True, startangle=0)
plt.title('Pre Model')

plt.subplot(1,4,2)
plt.pie(size2, explode=explode, labels=labels, autopct='%.2f%%',textprops={'fontsize':13}, shadow=True, startangle=0)
plt.title('RtO Model')


plt.subplot(1,4,3)
#plt.pie(size3, explode=explode, labels=labels, autopct='%.2f%%',textprops={'fontsize':14}, shadow=True, startangle=0)
plt.pie(size3, explode=explode, labels=labels, textprops={'fontsize':14}, shadow=True, startangle=0)
plt.title('RtN Model')
lbs = list()
for i in range(5):
    lbs.append(labels[i]+'('+str(size3[i])+'%)')
plt.legend(lbs,loc='best')


plt.subplot(1,4,4)
#plt.pie(size4, explode=explode, labels=labels, autopct='%.2f%%',textprops={'fontsize':14}, shadow=True, startangle=0)
plt.pie(size4, explode=explode, labels=labels,textprops={'fontsize':14}, shadow=True, startangle=0)
lbs = list()
for i in range(5):
    lbs.append(labels[i]+'('+str(size4[i])+'%)')
plt.legend(lbs,loc='best')
plt.title('RtM Model')



plt.show()