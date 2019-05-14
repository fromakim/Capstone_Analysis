# %% In[0]: Import Package
import cv2

# %% In[1]: Open IMG
heat = cv2.imread('./20190507_191615.bmp');
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(heat)

cv2.imshow('image', h)
cv2.waitKey(0)
cv2.destroyAllWindows()
