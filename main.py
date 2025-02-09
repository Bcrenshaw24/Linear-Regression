import pandas as pd 
import matplotlib.pyplot as plt 
 
data = pd.read_csv('data.csv') 

def compute_loss(m, b, points):
    n = len(points)
    total_loss = 0
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        prediction = m * x + b
        total_loss += (y - prediction) ** 2
    
    mse = total_loss / n
    return mse 

#cm = current weight value
def gradient(cm, b_now, points, L):  
    #initiates gradient variables to 0 
    mg, bg = 0, 0
 
    #To calculate the mean squared error
    n = len(points)  

    for i in range(n): 
        x = points.iloc[i].studytime
        y = points.iloc[i].score


        #Machine learning to move opposite of steepest ascent 
        mg += -(2/n) * x * (y - (cm * x + b_now))
        bg += -(2/n) * (y - (cm * x + b_now))

    #adjusts weights and bias
    m = cm - mg * L
    b = b_now - bg * L 
    return m, b 
#random values 
m = 2 
b = 30
L = 0.001
epochs = 1500

#Training the linear regression model
for i in range(epochs): 
    if i % 100 == 0: 
        print(f"Epoch: {i}, m: {m}, b: {b}") 
        print(compute_loss(m, b, data))
    m, b = gradient(m, b, data, L) 
 
print(f"m: {m} b: {b}") 
 
plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(0, 50)), [m * x + b for x in range(0, 50)], color="red") 
plt.xlabel("study time")
plt.ylabel("score")  
plt.show()




