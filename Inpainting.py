from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import threading
import gc
from Tips import *
#############################################
InpaintingPipe0 = None
InpaintingPipe1 = None
#############################################
def LoadInpaintingModel0():
    global InpaintingPipe0
    InpaintingPipe0=StableDiffusionInpaintPipeline.from_pretrained(
            "Safetensors/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
    InpaintingPipe0.to("cuda")
    
def LoadInpaintingModel1():
    global InpaintingPipe1
    InpaintingPipe1=StableDiffusionInpaintPipeline.from_pretrained(
            "Safetensors/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
    InpaintingPipe1.to("cuda")
    InpaintingPipe1.image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            "Safetensors/CLIP-ViT-H-14-laion2B-s32B-b79K",  # ✅ 用你本地路径
            local_files_only=True,                 # 强制只用本地文件，避免联网干扰
            torch_dtype=torch.float16
        ).to("cuda")
    InpaintingPipe1.load_ip_adapter(
        pretrained_model_name_or_path_or_dict="Safetensors/ip_adapter_plus_sd15", 
        weight_name="ip-adapter-plus_sd15.safetensors",
        subfolder=""
    )
    InpaintingPipe1.set_ip_adapter_scale(0.7)
################################################
def Inpainting(image:Image.Image,mask:Image.Image,prompt:str,negative_prompt_list:list=None):
    #
    negative_prompt=""
    if negative_prompt_list!=None:
        for x in negative_prompt_list:
            negative_prompt+=x+' '
    else:
        negative_prompt=None
    #  
    global InpaintingPipe0 
    if InpaintingPipe0 is None:
        LoadInpaintingModel0()
    res = InpaintingPipe0(prompt=prompt, image=image, mask_image=mask,height=1024,width=1024,negative_prompt=negative_prompt).images[0]
    #卸载模型
    '''
    del InpaintingPipe0
    gc.collect()
    torch.cuda.empty_cache()
    InpaintingPipe0=None
    '''
    return res.resize(image.size)

def InpaintingByIpAdapter(image:Image.Image,mask:Image.Image,prompt:str,adapter_img:Image.Image,negative_prompt_list:list=None):
    #
    negative_prompt=""
    if negative_prompt_list!=None:
        for x in negative_prompt_list:
            negative_prompt+=x+' '
    else:
        negative_prompt=None
    #
    LoadInpaintingModel1()
    global InpaintingPipe1
    res = InpaintingPipe1(prompt=prompt, image=image, mask_image=mask,height=1024,width=1024,ip_adapter_image=adapter_img,negative_prompt=negative_prompt).images[0]
    #卸载模型
    del InpaintingPipe1
    gc.collect()
    torch.cuda.empty_cache()
    #
    InpaintingPipe1=None
    return res.resize(image.size)
#######################
if __name__=="__main__":
    idx=0
    while True:
        img=Image.open(input("请输入图片:")).convert("RGB").resize((1024,1024))
        mask = Image.new("L", (1024, 1024), 0)
        draw_mask = np.array(mask)
        draw_mask[:, :] = 255 
        mask_image = Image.fromarray(draw_mask)
        des=input("请输入提示词:")
        res=Inpainting(img,mask,des)
        res.save(f"{idx}.png")
        idx+=1
        