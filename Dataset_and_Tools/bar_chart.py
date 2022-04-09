from turtle import color
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,6))
colors = ('#4287f5', '#4287f5', '#4287f5', '#f24741', '#4287f5', '#4287f5')
networks = ['2012 \n AlexNet','2013 \n ZFNet','2014 \n GoogLeNet','Human','2015 \n ResNet','2016 \n Trimps Soushen']
errors = [16.4, 11.7, 6.7, 5, 3.6, 3]
values = ['16.4%', '11.7%', '6.7%', '5%', '3.6%', '3%']

bars = ax.bar(networks, errors, color=colors)
ax.bar_label(bars,labels=values)

plt.title('Top-5 Error Rate of the ILSVRC winner from 2012 to 2016')
plt.xlabel('Year and the winner network')
plt.ylabel('Top-5 error rate (%)')

plt.show()