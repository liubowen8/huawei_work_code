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
