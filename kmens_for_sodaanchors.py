"""
通过使用k means的方法，产生soda数据集中anchor的size
"""

from tqdm import tqdm
import json, math
import matplotlib.pyplot as plt
import numpy as np

annotations_json_path="/home3/hqlab/liubowen/data/SSLAD-2D/labeled/annotations/"


def get_anchons( annotations_json_path, train_val='train'):
    
    json_path=annotations_json_path+"instance_"+train_val+".json"
    json_f=open(json_path,'r')
    json_dict=json.load(json_f)
    annotations=json_dict["annotations"]  # list 里面是每一个object的标注信息，一个对象使用一个字典标
    # 生成label文件夹和 其下的txt文件
    whs=[]
    for object_info in tqdm(annotations):
        #object_info是一个字典
        image_id=object_info["image_id"]
        #"category_id": 3, "bbox": [65, 667, 174, 126], "area": 21924, "id": 1, "iscrowd": 0
        category_id=object_info["category_id"]
        bbox=object_info["bbox"]
        xcenter, ycenter, w, h= bbox
        wh=[w,h]
        whs.append(wh)
    #whs=np.array(whs)
    return whs
    
    



def main():
    whs=get_anchons(annotations_json_path , "train")
    points_of_centers=[[],[],[],[],[],[]]
    centers=[[174,126],[30, 52],[27, 44],[208,111],[0, 0],[70, 123]]
    while True:
        print(np.array(centers).astype(dtype=int).tolist(),1111)
        for point in whs:
            distences=[]
            for center in centers:
                distance=math.sqrt((point[0]-center[0])**2+(point[1]-center[1])**2)
                distences.append(distance)
            index=distences.index(min(distences))
            points_of_centers[index].append(point)
        next_centers=[]
        for points_of_centeri in points_of_centers:
            x=np.array(points_of_centeri)
            xx=np.mean(x, axis=0)
            center=list(xx)
            next_centers.append(center)
        if np.array(centers).astype(dtype=int).tolist()==np.array(next_centers).astype(dtype=int).tolist():
            print("find:" ,np.array(next_centers).astype(dtype=int).tolist())
            break
        else :
            centers=next_centers
            
 


if __name__ == "__main__":
    main()
