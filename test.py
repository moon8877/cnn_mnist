import torch
from cnn import CNN
import cv2
import torchvision
cnn3 = torch.load('net.pkl')
img = cv2.imread('./8.png')
transforms2 = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
new_img = cv2.resize(img,(28,28))
new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
#new_img = new_img
#cv2.imshow('aa',new_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img_cv_tensor = transforms2(new_img)
img_cv_tensor = torch.squeeze(img_cv_tensor,dim=0)
img_cv_tensor = torch.unsqueeze(img_cv_tensor,dim=0)
img_cv_tensor = torch.unsqueeze(img_cv_tensor,dim=0)
newout = cnn3(img_cv_tensor)
pred_y = torch.max(newout, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')