import torch
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM



class SwitchCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(SwitchCAM, self).__init__(model, target_layer, use_cuda, 
            reshape_transform=reshape_transform)
        self.model = model

    def minMax(self,tensor):
        maxs = tensor.max(dim=1)[0]
        mins = tensor.min(dim=1)[0]
        return (tensor-mins)/(maxs-mins)

    def get_cos_similar_matrix(self,v1, v2):
        num = np.dot(v1, np.array(v2).T) 
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
        res = num / denom
        res[np.isnan(res)] = 0
        return res
    
    def combination(self,scores,grads_tensor,factor=3):
        scores = self.minMax(scores)
        grads = self.minMax(grads_tensor)
        grads_class = 2.0*grads - 1.0
        weight = grads
        weights = grads_class
        factor_prod = 1.0

        for i in range(factor):
            factor_prod = (i+2)*factor_prod
            weight *= scores
            weights += 2.0*self.minMax(weight)/factor_prod

        return weights

    def get_cam_weights(self,input_tensor,target_category,activations,grads):
        with torch.no_grad():
            grads_tensor = np.mean(grads, axis=(2, 3))
            activation = activations.reshape(activations.shape[1],-1)
            consine = self.get_cos_similar_matrix(activation, activation)
 

            activation_tensor = torch.from_numpy(activations)
            consine = torch.from_numpy(consine)
            activation = torch.from_numpy(activation)

            record = torch.zeros(consine.shape)

            for i in range(consine.shape[0]):
                threshold0 = torch.quantile(consine[i,:],0.95)
                record[i,:] = consine[i,:]>threshold0



            if self.cuda:
                activation_tensor = activation_tensor.cuda()
                grads_tensor = torch.Tensor(grads_tensor).cuda()

            scores = torch.zeros(activation_tensor.shape[1]).cuda() 
            orig_result = self.model.classifier(torch.flatten(activation_tensor,1))[:,target_category]



            for i in range(activation_tensor.shape[1]):
                if scores[i] != 0.0:
                    continue
                switch = torch.zeros(activation_tensor.shape).cuda()

                switch[:,torch.where(record[i])[0].numpy(),:,:]=activation_tensor[:,torch.where(record[i])[0].numpy(),:,:]
                flatten = torch.flatten(switch,1)
                score = self.model.classifier(flatten)[:,target_category]
                flatten = torch.flatten(activation_tensor - switch,1)
                score = orig_result - self.model.classifier(flatten)[:,target_category]+score
                scores[torch.where(record[i])[0].numpy()] = score

            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = self.combination(scores,grads_tensor,factor=3)
            
            return weights.cpu().numpy()