from LLM import *
from VLM import *
import threading
import ast
################################
#获取任务
def GetTask(image,description:str,API=True):
    answer=AnswerImage([image],Expert1_Prompt,f"My task is:{description}",API)
    #对answer细分
    try:
        lst=ast.literal_eval(answer)
        #lst=json.loads(answer)
        if type(lst)!=list:
            raise Exception("type is not list")
        return lst
    except Exception as e:
        Debug("GetTask:",e," ",answer)
        return GetTask(image,description,API=API)
#获取编辑后的全局打分
def GetImageGlobalScore(source,target,description:str):
    res=GetImageScore([source,target],GlobalScore_Prompt,"The image is as above, and my editing instruction for this round is {}".format(description))
    cost=Timer()
    pos_prompt=""
    neg_prompt=""
    if len(res[1])!=0:
        pos_prompt=SummaryPrompt(res[1])
    if len(res[2])!=0:
        neg_prompt=SummaryPrompt(res[2])
    Debug("SummaryPrompt cost:",cost())
    return res[0],pos_prompt,neg_prompt
#艺术家打分
def GetCriticScore(source,target,instruction:str):
    question=f'''
        Now I am giving my images, before editing and after editing, as shown above.
        All the editing commands are as follows {instruction}
    '''
    return AnswerImage([source,target],Critic_Prompt,question) 
#指令优化
def OptmEditInstruction(prompt:str,instruction:str):
    tip=f'''
        Now I will give my query:.
        positive prompt word:{prompt}
        Instructions for this round of editing:{instruction}
    '''
    res=AnswerText(InsOptim_Prompt,tip)
    try:
        data=json.loads(res)
        return data["new_instruction"]
    except Exception as e:
        Debug("OptmEditInstruction_Err:",res,e)
        return ""
#对反馈进行总结
def SummaryPrompt(prompts:list)->str:
    question=f'''
        Now I give my prompts:{str(prompts)}
    '''
    res=AnswerText(PromptFeedbackSummary_Prompt,question)
    try:
        data=json.loads(res)
        return data["prompt"]
    except Exception as e:
        Debug("SummaryPrompt:",e,res)
        return SummaryPrompt(prompts)
#获取prompt的操作区域和生成目标
def GetSoureAndTargetArea(image:Image.Image,prompt:str):
    client=client1()
    response = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="qwen-flash",
        messages=[
                {
                    "role":"system",
                    "content": [
                            {"type": "text", "text": f"{DividePrompt_Prompt}"},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                            {"type": "text", "text": f"{prompt}"},
                    ]+
                    [ {"type": "image_url", "image_url": {"url": encode_image(image.resize((512,512)))}}]
                    ,
                },
            ],
        extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920
        },
    )
    Debug("reason:",response.choices[0].message.reasoning_content)
    res=response.choices[0].message.content
    #res=AnswerImage([image],DividePrompt_Prompt,prompt)
    try:
        data=json.loads(res)
        source=data["source"]
        source=source if source!="none" else None
        target=data["target"]
        target=target if target!="none" else None
        mask=data["mask"]
        w,h=image.size
        return source,target,[int(mask[0]*w),int(mask[1]*h),int(mask[2]*w),int(mask[3]*h)]
    except Exception as e:
        Debug("GetSoureAndTargetArea:",e,res)
        return GetSoureAndTargetArea(image,prompt)
##########################################
if __name__=="__main__":
    basedir="data/vincie/"
    foler=[]
    with open("7.txt","r")as f:
        for line in f:
            foler.append(line.strip())
    import random
    random.shuffle(foler)
    with open("log.txt","a") as out:
        for x in foler:
            print("current is {}".format(x))
            p=f"{basedir}/{x}"
            task=""
            with open(f"{p}/task.txt","r")as f:
                task=f.read()
            src=Image.open(f"{p}/src.jpg").convert("RGB")
            task=GetTask(src,task,False)
            if len(task)>=4:
                out.write(str(x)+'\n')
                out.flush()
                with open(f"{p}/task.txt","w") as f:
                    f.write(str(task))
                print(task)
        