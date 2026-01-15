from Tips import *
from ImageEdit import *
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
#########################使用vincie数据集测试
def PredictByCase():
    all=[]
    for i in range(1,5):
        dir=f"case"
        with open(f"{dir}/task{i}.txt","r",encoding="utf-8") as f:
            task=f.read()
        all.append({"task":task,"input_img":f"{dir}/{i}.jpg","dir":f"debug/{i}"})
    return all
def PredictByVincie():
    all=[]
    with open("log.txt","r")as f:
        for x in f:
            all.append(int(x.strip()))
    all=set(all)
    all=list(all)
    random.shuffle(all)
    ret=[]
    for x in all:
        dir=f"data/vincie/{x}"
        with open(f"{dir}/task.txt","r",encoding="utf-8") as f:
            task=f.read()
        ret.append({"task":task,"input_img":f"{dir}/src.jpg","dir":f"debug/{x}"})        
    return ret
def PredictByMagicBrush():
    #获取所有待测试的数据
    target={}
    dir="data/MagicBrush"
    for folder in os.listdir(dir):
        path=f"{dir}/{folder}"
        cnt=len(os.listdir(path))//3
        target_img=f"{path}/source_1.png"
        run_dir=f"debug/{folder}"
        task=[]
        for i in range(1,cnt+1):
            with open(f"{path}/task_{i}.txt","r",encoding="utf-8") as f:
                t=f.read()
            task.append(t)
        if cnt not in target:
            target[cnt]=[]
        target[cnt].append({"input_img":target_img,"task":task,"dir":run_dir})
    #
    target=target[3]
    #随机选择3轮次的进行编辑
    ret=[]
    cnt=min(TEST_CNT,len(target))
    for i in range(cnt):
        if len(target)==0:
            break
        idx=random.randint(0,len(target)-1)
        ret.append(target[idx])
        target=target[:idx]+target[idx+1:]
    return ret
    
###########################使用NanoBanana测试
def PredictByNanoBanana():
    dir="Nano-150k"
    all=[]
    for folder in os.listdir(f"{dir}/json/"):
        with open(f"{dir}/json/{folder}","r",encoding="utf-8") as f:
            data=f.read()
        data=data.split("\n")
        for line in data:
            try:
                dt=json.loads(line)
                if len(dt["input_images"])==1:
                    all.append({"task":dt["instruction"],"input":dt["input_images"][0],"output":dt["output_image"]}) 
            except Exception as e:
                pass
    #随机抽取
    ret=[]
    cnt=min(TEST_CNT,len(all))
    for i in range(cnt):
        idx=random.randint(0,len(all)-1)
        target=all[idx]
        #创建目录
        dir=f"{DEBUG_DIR}/{idx}/"
        ret.append({"task":target["task"],"input_img":target["input"],"dir":dir})
    return ret
#################################
if __name__=="__main__":
    res=PredictByVincie()
    print(res)