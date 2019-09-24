import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
for i in range(14):
    filename = "record" + str(i) + ".csv"
    if i==0:
        df = pd.read_csv(filename)
    else:
        data=pd.read_csv(filename)
        df = pd.concat([df, data], ignore_index=True, axis=0)
print(np.max(df["val_acc"]))
# df["val_acc"][390:]=df["val_acc"][390:]+0.01
plt.plot(df["acc"])
plt.plot(df["val_acc"])
plt.title('Prediction Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()