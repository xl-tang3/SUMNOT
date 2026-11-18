import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch.distributed as dist
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

def myprint(message, local_rank=0):
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(message)
    else:
        print(message)

class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.D = Degradation(args)
        self.de_type = self.args.de_type 
        
        # 1. 自动识别 simulation 任务 (带下划线的，如 denoise_40)
        self.sim_tasks = [x for x in self.de_type if '_' in x] 
        
        # 2. 自动识别平行文件夹任务 (single)
        # [修复]: 不要在这里直接 listdir，防止路径不存在导致报错。移到 _init_ids 处理。
        self.folder_tasks = [] 

        self.sample_ids = []
        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

    def _init_ids(self):
        # --- A. Simulation 模式 (如 denoise_XX) ---
        if self.sim_tasks:
            # 这里假设所有模拟任务都共享同一份 clean images (来自 denoise_dir)
            # 如果不同任务源不同，需要进一步根据 task 类型区分逻辑
            ref_file = os.path.join(self.args.data_file_dir, "noisy/denoise.txt")
            if os.path.exists(ref_file) and os.path.exists(self.args.denoise_dir):
                temp_ids = [id_.strip() for id_ in open(ref_file)]
                name_list = os.listdir(self.args.denoise_dir)
                clean_ids = [os.path.join(self.args.denoise_dir, id_) for id_ in name_list if id_.strip() in temp_ids]
                
                for x in clean_ids:
                    self.sample_ids.append({
                        "clean_id": x, 
                        "mode": "simulation", 
                        "tasks": self.sim_tasks 
                    })
                myprint(f"Simulation Mode: {len(self.sim_tasks)} versions per image. Total: {len(clean_ids)}")
            else:
                myprint(f"Warning: Simulation tasks found {self.sim_tasks} but data path invalid.")

        # --- B. Folders 模式 (single) ---
        if 'single' in self.de_type:
            if os.path.exists(self.args.single_dir):
                # [修复]: 在确认路径存在且需要时才读取 level 文件夹
                self.folder_tasks = [x for x in os.listdir(self.args.single_dir) if x.startswith('level')]
                self.folder_tasks.sort() # 保证 level1, level2... 顺序
                
                if self.folder_tasks:
                    base_dir = os.path.join(self.args.single_dir, self.folder_tasks[0])
                    image_list = os.listdir(base_dir)
                    for x in image_list:
                        self.sample_ids.append({
                            "clean_id": x, 
                            "mode": "folders", 
                            "tasks": self.folder_tasks
                        })
                    myprint(f"Folder Mode: {len(self.folder_tasks)} levels per image. Total: {len(image_list)}")
                else:
                    myprint("Warning: 'single' mode set but no 'level*' folders found in single_dir.")
            else:
                myprint(f"Warning: 'single' mode set but path {self.args.single_dir} does not exist.")
        
        random.shuffle(self.sample_ids)

    # [修复]: 补充缺失的 _sync_crop 方法
    def _sync_crop(self, img_list):
        """对一组图片进行完全相同位置的裁剪"""
        H, W, _ = img_list[0].shape
        p = self.args.patch_size
        # 防止图片小于 patch_size
        if H < p or W < p:
             # 简单 resize 或者 pad，这里直接抛出异常或做兼容处理
             # 实际项目中通常建议 resize 或者 pad
             pass 
        
        ind_H = random.randint(0, H - p)
        ind_W = random.randint(0, W - p)
        return [img[ind_H:ind_H + p, ind_W:ind_W + p, :] for img in img_list]

    # [修复]: 补充缺失的 _sync_aug 方法
    def _sync_aug(self, img_list):
        """对一组图片进行完全相同方式的数据增强"""
        # 生成一个随机模式 0-7 (需确保 utils.image_utils.random_augmentation 支持 mode 参数)
        # 如果 random_augmentation 不支持 mode，可以手动实现翻转/旋转
        mode = random.randint(0, 7)
        return [random_augmentation(img, mode)[0] for img in img_list]

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        
        if sample["mode"] == "simulation":
            # --- 动态模拟 N 个版本 ---
            clean_id = sample["clean_id"]
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            
            # 先裁剪
            clean_patch = np.array(self.crop_transform(clean_img))
            # 再增强
            clean_patch = random_augmentation(clean_patch)[0]
            
            degrad_patches = []
            de_params = []
            
            # for task_str in sample["tasks"]:
            #     # 解析任务，如 "denoise_40"
            #     if '_' in task_str:
            #         task_type, param = task_str.split('_')
            #         param = int(param)
            #     else:
            #         task_type = task_str
            #         param = 0
                
            #     # 调用 Degradation 类
            #     if task_type == 'denoise':
            #         p = self.D.add_gaussian_noise(clean_patch, param)
            #     # 可扩展其他类型
            #     # elif task_type == 'deblur':
            #     #     p = self.D.add_blur(clean_patch, param)
            #     else:
            #         p = clean_patch 
                
            #     degrad_patches.append(p)
            #     de_params.append(param)
                
            # clean_name = clean_id.split("/")[-1].split('.')[0]
# ------------------------------------------------------------------------------------


            import random
            # -----------------------------------------------------------
            # Step A: Pre-scan to find the dynamic range of noise levels
            # -----------------------------------------------------------
            # Extract all sigma values from tasks like "denoise_15", "denoise_50"
            denoise_candidates = [
                int(t.split('_')[1]) for t in sample["tasks"] 
                if t.startswith('denoise_') and '_' in t
            ]

            # Determine the min and max sigma for random sampling
            if denoise_candidates:
                min_sigma = min(denoise_candidates)
                max_sigma = max(denoise_candidates)
            else:
                # Fallback defaults if no denoise tasks exist
                min_sigma, max_sigma = 15, 50

            # -----------------------------------------------------------
            # Step B: Main Loop Processing
            # -----------------------------------------------------------
            for task_str in sample["tasks"]:
                # Parse task string, e.g., "denoise_40" -> type="denoise", param=40
                if '_' in task_str:
                    task_type, param = task_str.split('_')
                    param = int(param)
                else:
                    task_type = task_str
                    param = 0
                
                # Initialize final_sigma to track what is actually applied
                applied_param = param 

                # Call Degradation Class
                if task_type == 'denoise':
                    # Configuration: 30% chance to use random continuous noise
                    prob_random = 0.3
                    
                    if random.random() < prob_random:
                        # Random Mode: Sample a float between [min_sigma, max_sigma]
                        # This fills the gaps between fixed anchors (e.g., 33.5, 18.2)
                        final_sigma = random.uniform(min_sigma, max_sigma)
                    else:
                        # Anchor Mode: Use the specific fixed param (e.g., 15, 25, 50)
                        final_sigma = float(param)


                    
                    # Apply noise
                    p = self.D.add_gaussian_noise(clean_patch, final_sigma)
                    
                    # Update the applied parameter to reflect the actual noise level
                    # (Useful if your network takes this value as conditioning input)
                    applied_param = final_sigma

                # Extend for other types if needed
                # elif task_type == 'deblur':
                #     p = self.D.add_blur(clean_patch, param)
                else:
                    p = clean_patch 
                
                degrad_patches.append(p)
                # Append the actual parameter used (might be float now)
                de_params.append(applied_param)
                
            clean_name = clean_id.split("/")[-1].split('.')[0]
        
        else:
            # --- 动态读取 N 个文件夹 ---
            img_name = sample["clean_id"]
            # 读取 GT
            clean_path = os.path.join(self.args.single_dir, 'target', img_name)
            clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
            
            # 读取所有退化层级
            de_imgs = []
            for lv_dir in sample["tasks"]:
                lv_path = os.path.join(self.args.single_dir, lv_dir, img_name)
                de_imgs.append(crop_img(np.array(Image.open(lv_path).convert('RGB')), base=16))
            
            # [修复]: 调用补充好的 _sync_crop 和 _sync_aug
            all_imgs = self._sync_aug(self._sync_crop([clean_img] + de_imgs))
            
            clean_patch = all_imgs[0]
            degrad_patches = all_imgs[1:]
            
            # 文件夹模式用索引作为 ID
            de_params = list(range(len(degrad_patches))) 
            clean_name = img_name

        # Final conversion
        clean_tensor = self.toTensor(clean_patch)
        # 自动堆叠: [N, C, H, W]
        degrad_tensor_stacked = torch.stack([self.toTensor(p) for p in degrad_patches], dim=0)

        return [clean_name, torch.tensor(de_params)], degrad_tensor_stacked, clean_tensor

    def __len__(self):
        return len(self.sample_ids)

# =============================================================================
# 训练数据集：MFMRI_TrainDataset
# 功能：
# 1. 读取 1 张 HF (320x320) 和 3 张 LF (128/192/256)
# 2. 将所有 LF 强制 Resize 到 HF 的尺寸 (320x320)
# 3. 对 4 张图进行完全同步的 RandomCrop 和 Augmentation
# 4. 返回：文件名, LF_Stack [3, C, H, W], HF_Tensor [C, H, W]
# =============================================================================
class MFMRI_TrainDataset(Dataset):
    def __init__(self, args):
        super(MFMRI_TrainDataset, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        
        # 1. 确定根目录
        if hasattr(args, 'data_dir'):
            self.root = args.data_dir
        elif hasattr(args, 'data_file_dir'):
            self.root = args.data_file_dir
        else:
            raise ValueError("Args missing 'data_dir' or 'data_file_dir'")
        
        self.target_folder = 'HF'
        
        # =========================================================
        # [核心修改]：混合模式 (自动扫描 OR 手动指定)
        # =========================================================
        
        # 假设 args.de_type 是一个列表，例如 ['all'] 或 ['LF_mild', 'LF_Severe']
        # 这一步是为了兼容 parser nargs='+'
        de_type_list = args.de_type if isinstance(args.de_type, list) else [args.de_type]

        if 'all' in de_type_list:
            # --- 模式 A: 自动扫描所有 LF_ 开头的文件夹 ---
            all_items = os.listdir(self.root)
            self.source_folders = [
                d for d in all_items 
                if d.startswith('LF_') and os.path.isdir(os.path.join(self.root, d))
            ]
            self.source_folders.sort() # 必须排序，保证 DDP 一致性
            
            if len(self.source_folders) == 0:
                raise ValueError(f"在 {self.root} 下没有找到任何 'LF_*' 文件夹！")
                
            print(f"[Dataset] 自动模式 (Auto): 扫描到 {len(self.source_folders)} 个源 -> {self.source_folders}")
            
        else:
            # --- 模式 B: 手动指定 (如只练 mild 和 severe) ---
            self.source_folders = de_type_list
            self.source_folders.sort() # 依然建议排序，确保 Domain ID (0,1,2...) 固定
            
            # 简单检查一下文件夹是否存在，防止写错名
            for folder in self.source_folders:
                full_path = os.path.join(self.root, folder)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"手动指定的文件夹不存在: {full_path}")
            
            print(f"[Dataset] 手动模式 (Manual): 使用指定的 {len(self.source_folders)} 个源 -> {self.source_folders}")

        # =========================================================

        # 4. 扫描 HF/train 获取文件名列表
        target_path = os.path.join(self.root, self.target_folder, 'train')
        if not os.path.exists(target_path):
             raise FileNotFoundError(f"找不到 Target 路径: {target_path}")

        self.file_names = [x for x in os.listdir(target_path) if x.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        print(f"[Dataset] 已加载 {len(self.file_names)} 组样本。")

    def _sync_crop(self, img_list):
        H, W, _ = img_list[0].shape
        p = self.args.patch_size
        if H < p or W < p: return img_list
        ind_H = random.randint(0, H - p)
        ind_W = random.randint(0, W - p)
        return [img[ind_H:ind_H + p, ind_W:ind_W + p, :] for img in img_list]

    def _sync_aug(self, img_list):
        if random.random() < 0.5: img_list = [np.fliplr(img).copy() for img in img_list]
        if random.random() < 0.5: img_list = [np.flipud(img).copy() for img in img_list]
        if random.random() < 0.5: 
            k = random.randint(1, 3)
            img_list = [np.rot90(img, k).copy() for img in img_list]
        return img_list

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # 1. 读取 Target (HF) -> 获取基准尺寸
        hf_path = os.path.join(self.root, self.target_folder, 'train', file_name)
        hf_img = Image.open(hf_path).convert('RGB')
        W_base, H_base = hf_img.size

        # 2. 动态读取 self.source_folders (无论是自动扫的还是手动定的)
        lf_imgs = []
        for folder in self.source_folders:
            path = os.path.join(self.root, folder, 'train', file_name)
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺失文件: {path}")
                
            img = Image.open(path).convert('RGB')
            
            # Resize 对齐
            if img.size != (W_base, H_base):
                img = img.resize((W_base, H_base), resample=Image.BICUBIC)
            
            lf_imgs.append(np.array(img))

        hf_img = np.array(hf_img)

        # 3. 同步处理
        combined = [hf_img] + lf_imgs
        combined = self._sync_crop(combined)
        combined = self._sync_aug(combined)

        # 4. 堆叠返回
        clean_tensor = self.toTensor(combined[0]) # HF
        # 堆叠 K 张 LF 图
        degrad_stack = torch.stack([self.toTensor(p) for p in combined[1:]], dim=0)

        return file_name, degrad_stack, clean_tensor

    def __len__(self):
        return len(self.file_names)

class MFMRI_TestDataset(Dataset):
    def __init__(self, args, level='LF_0.055T'):
        super(MFMRI_TestDataset, self).__init__()
        self.toTensor = ToTensor()
        self.args = args
        
        if hasattr(args, 'data_dir'): self.root = args.data_dir
        elif hasattr(args, 'data_file_dir'): self.root = args.data_file_dir
        else: raise ValueError("Args error")

        self.input_folder = os.path.join(self.root, level, 'test')
        self.target_folder = os.path.join(self.root, 'HF', 'test')
        
        if os.path.exists(self.input_folder):
            self.file_names = [x for x in os.listdir(self.input_folder) if x.endswith(('.png', '.jpg'))]
        else:
            self.file_names = []
            print(f"[Warning] Path not found: {self.input_folder}")

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img_in = Image.open(os.path.join(self.input_folder, fname)).convert('RGB')
        img_gt = Image.open(os.path.join(self.target_folder, fname)).convert('RGB')
        if img_in.size != img_gt.size:
            img_in = img_in.resize(img_gt.size, Image.BICUBIC)
        return fname, self.toTensor(img_in), self.toTensor(img_gt)

    def __len__(self):
        return len(self.file_names)

class OmniTrainDataset(Dataset):
    def __init__(self, args, noise_combine=False):
        super(OmniTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        myprint(f"training degradation types, including: {self.de_type}")

        self.noise_combine = noise_combine
        if not noise_combine:
            self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5,
                            'lowlight': 6, 'single': 7}
            myprint(f"degradation type labels: {self.de_dict}")
        else:
            self.de_dict_combine = {'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2,
                                    'deblur': 3, 'lowlight': 4}
            myprint(self.de_dict_combine)

        self._init_ids()
        self._merge_ids()
        
        # Initialize Unpaired B Domain
        self._init_unpaired_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            myprint('... init denoising train dataset')
            self._init_noise_clean_ids()
        if 'derain' in self.de_type:
            myprint('... init derain train dataset')
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            myprint('... init dehazing train dataset')
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            myprint('... init debluring train dataset')
            self._init_blur_ids()
        if 'lowlight' in self.de_type:
            myprint('... init lowlight train dataset')
            self._init_lowlight_ids()
        if 'single' in self.de_type:
            myprint('... init single train dataset')
            self._init_single_ids()
        random.shuffle(self.de_type)

    def _init_unpaired_ids(self):
        myprint('... init unpaired LQ/HQ target B dataset')
        valid_exts = ('.png', '.jpg', '.jpeg')
        self.unpaired_paths = [
            os.path.join(self.args.unpaired_dir, f) for f in os.listdir(self.args.unpaired_dir) 
            if f.lower().endswith(valid_exts)
        ]
        self.unpaired_lq_paths = [
            os.path.join(self.args.unpaired_lq_dir, f) for f in os.listdir(self.args.unpaired_lq_dir)
            if f.lower().endswith(valid_exts)
        ]
        self.num_unpaired = len(self.unpaired_paths)
        self.num_unpaired_lq = len(self.unpaired_lq_paths)
        if self.num_unpaired == 0:
            raise ValueError(f"No HQ unpaired images found in {self.args.unpaired_dir}")
        if self.num_unpaired_lq == 0:
            raise ValueError(f"No LQ unpaired images found in {self.args.unpaired_lq_dir}")
        myprint("Total Unpaired HQ Ids : {}".format(self.num_unpaired))
        myprint("Total Unpaired LQ Ids : {}".format(self.num_unpaired_lq))

    def _init_noise_clean_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "noisy/denoise.txt")
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]
        self.s15_ids = self.s25_ids = self.s50_ids = []
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 5
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25'  in self.de_type and 'denoise_15' in self.de_type and 'denoise_50' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 5
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        elif  'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 15
            random.shuffle(self.s25_ids)
            self.s25_counter = 0            
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 5
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_s50 = len(self.s50_ids)
        self.num_s25 = len(self.s25_ids)
        self.num_s15 = len(self.s15_ids)

        self.num_clean =  self.num_s50 + self.num_s25 + self.num_s15
        myprint("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = os.path.join(self.args.data_file_dir, "hazy/hazy_outside.txt")
        temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        myprint("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = os.path.join(self.args.data_file_dir, "rainy/rainTrain.txt")
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 360
        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        myprint("Total Rainy Ids : {}".format(self.num_rl))

    def _init_blur_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.deblur_dir, 'sharp/'))
        temp_ids = image_list
        self.blur_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]
        self.blur_ids = self.blur_ids * 35 
        self.blur_counter = 0
        self.num_blur = len(self.blur_ids)
        myprint("Total Blur Ids : {}".format(self.num_blur))

    @staticmethod
    def _check_blur_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

    def _init_lowlight_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.lowlight_dir, 'low/'))
        temp_ids = image_list
        self.lowlight_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids]
        self.lowlight_ids = self.lowlight_ids * 150
        self.num_lowlight = len(self.lowlight_ids)
        myprint("Total low light Ids : {}".format(self.num_lowlight))

    def _init_single_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.single_dir, 'degraded/'))
        temp_ids = image_list
        self.single_ids = [{"clean_id": x, "de_type": 7} for x in temp_ids]
        self.single_ids = self.single_ids * 5
        self.num_single = len(self.single_ids)
        myprint("Total single Ids : {}".format(self.num_single))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        return patch_1, patch_2

    def _get_raingt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:  
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.blur_ids
        if 'lowlight' in self.de_type:
            self.sample_ids += self.lowlight_ids
        if 'single' in self.de_type:
            self.sample_ids += self.single_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def _resample_ids(self, sample_num):
        raise NotImplementedError("not provided yet")

    def __getitem__(self, idx):
        # ---------------------------------------------------------
        # Domain A: Standard Paired Fetching
        # ---------------------------------------------------------
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id < 3:
            clean_id = sample["clean_id"]
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16) 
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)
            clean_name = clean_id.split("/")[-1].split('.')[0]
            clean_patch = random_augmentation(clean_patch)[0]
            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_raingt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.deblur_dir, 'blur/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.deblur_dir, 'sharp/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = sample["clean_id"]
            elif de_id == 6:
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.lowlight_dir, 'low/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.lowlight_dir, 'high/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = sample["clean_id"]
            elif de_id == 7:
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.single_dir, 'degraded/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.single_dir, 'target/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = sample["clean_id"]

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        if not self.noise_combine:
            de_id = de_id
        else:
            self.de_dict_combine = {'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2, 'deblur': 3, 'lowlight': 4}
            de_id -= 2
            if de_id < 0:
                de_id = 0

        batch_A = ([clean_name, de_id], degrad_patch, clean_patch)

        # ---------------------------------------------------------
        # Domain B: Independent Unpaired Target Fetching
        # ---------------------------------------------------------
        idx_lq = random.randint(0, self.num_unpaired_lq - 1)
        idx_hq = random.randint(0, self.num_unpaired - 1)
        lq_path = self.unpaired_lq_paths[idx_lq]
        hq_path = self.unpaired_paths[idx_hq]

        lq_img = crop_img(np.array(Image.open(lq_path).convert('RGB')), base=16)
        hq_img = crop_img(np.array(Image.open(hq_path).convert('RGB')), base=16)

        lq_patch = np.array(self.crop_transform(lq_img))
        hq_patch = np.array(self.crop_transform(hq_img))
        lq_patch = random_augmentation(lq_patch)[0]
        hq_patch = random_augmentation(hq_patch)[0]

        batch_B = ("unpaired", self.toTensor(lq_patch), self.toTensor(hq_patch))

        return batch_A, batch_B

    def __len__(self):
        return len(self.sample_ids)

    def get_num_samples(self):
        domain_sample_counts = []
        if 'denoise_25' in self.de_type or 'denoise_15' in self.de_type or 'denoise_50' in self.de_type:
            domain_sample_counts.append(self.num_clean)
        if 'derain' in self.de_type:
            domain_sample_counts.append(self.num_rl)
        if 'dehaze' in self.de_type:
            domain_sample_counts.append(self.num_hazy)
        if 'deblur' in self.de_type:
            domain_sample_counts.append(self.num_blur)
        if 'lowlight' in self.de_type:
            domain_sample_counts.append(self.num_lowlight)
        if 'single' in self.de_type:
            domain_sample_counts.append(self.num_single)

        return domain_sample_counts


