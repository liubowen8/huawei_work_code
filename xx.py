  loss_evi=loss_eq5(t,alpha,K_alpha).sum()/alpha.shape[0]*0.2   # *0.2 是为了和bceloss 值尽量相近
                #print(loss_evi,1111111)
                lcls +=  loss_evi
                
                
 def KL(alpha,K):
        beta=torch.ones(1,K).to(alpha.device)
        S_alpha=torch.sum(alpha, dim=1,keepdim=True)
        KL = torch.sum((alpha - beta)*(torch.digamma(alpha)-torch.digamma(S_alpha)),dim=1,keepdim=True) + \
            torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1,keepdim=True) + \
            torch.sum(torch.lgamma(beta),dim=1,keepdim=True) - torch.lgamma(torch.sum(beta,dim=1,keepdim=True))
        return KL

def loss_eq5(p, alpha, K):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum((p-(alpha/S))**2, dim=1, keepdim=True) + torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True)
    KL_reg =KL((alpha - 1)*(1-p) + 1 , K)
    return loglikelihood + KL_reg
  
  
def maximum_likelihood_loss(box1, box2, x1y1x2y2=True):
  # Returns the IoU of box1 to box2. box1 is 4*n, box2 is nx4
  box2 = box2.t()

  # Get the coordinates of bounding boxes
  if x1y1x2y2:  # x1, y1, x2, y2 = box1
      b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
      b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
  else:  # transform from xywh to xyxy
      b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
      b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
      b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
      b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
  sigma1=box1[4]
  sigma2=box1[5]  
  sigma3=box1[6]
  sigma4=box1[7] 
  #这里已经取了指数了
  #print(sigma1, sigma2,sigma3, sigma4,22222)
  loss1=torch.log(sigma1*sigma2)+0.5*(((b1_x1-b2_x1)/sigma1)**2+((b1_y1-b2_y1)/sigma2)**2)
  loss2=torch.log(sigma3*sigma4)+0.5*(((b1_x2-b2_x2)/sigma3)**2+((b1_y2-b2_y2)/sigma4)**2)
  loss=loss1.sigmoid()+loss2.sigmoid()
  return loss*0.5
