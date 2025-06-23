import torch
from CERTIFAI import CERTIFAI
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from utils import extract_numpy_from_loader, save_image, get_fixed_indices_from_mask, load_masks

class CertifaiMvTekWrapper:
    def __init__(self, dataset_dir:str, category:str, results_dir="./results/"):
        self.category = category
        self.dataset_dir = dataset_dir
        
        mask_dir = f"{dataset_dir}{category}/ground_truth/"
        self.mask_dir = mask_dir
        
        train_dir = f"{dataset_dir}{category}/train"
        val_dir = f"{dataset_dir}{category}/test"
        self.train_dir = train_dir
        self.val_dir = val_dir
        
        self.results_dir = results_dir
        self.isCertifAiInitialized = False
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        torch.set_float32_matmul_precision("medium")
        self.train_dataset = datasets.ImageFolder(train_dir ,transform=transform)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.X_train_np, self.y_train_np = extract_numpy_from_loader(self.train_loader)
        self.X_val_np, self.y_val_np = extract_numpy_from_loader(self.val_loader)

        self.shape = self.X_val_np.shape
        self.val_images, _, self.width, self.height = self.shape
        print(f"input image size is: {self.width}x{self.height}")
        
        mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.width, self.height)),
            transforms.ToTensor()
        ])
        
        self.val_masks = load_masks(self.val_dataset, mask_dir, mask_transform)
        
        torch.set_float32_matmul_precision("medium")
        
        self._initMvTec()
        self._trainMvTec()
    
    def _initMvTec(self):    
        self.datamodule = MVTec(
            root=self.dataset_dir,
            category=self.category,
            train_batch_size=16,
            eval_batch_size=16,
            num_workers=os.cpu_count()
        )
        
        self.model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
        self.engine = Engine()
        print(f"Model weight type: {next(self.model.parameters()).dtype}, device: {next(self.model.parameters()).device}")
        # model.to("cuda")
        
        self.datamodule.setup()
        
    def _trainMvTec(self):
        self.engine.fit(model=self.model, datamodule=self.datamodule)
        
    # Initialize with evolution for single sample
    def initCertifAi(self, sample_id_in_test_dataset):
        self.isCertifAiInitialized = True
        self.datamodule.setup()
        
        image_id = sample_id_in_test_dataset
        self.current_input_img_id = image_id
        self.cert = CERTIFAI(numpy_dataset=self.X_val_np[image_id:image_id+1])
        self.cert.set_distance("SSIM")
        self.maskIndicies = get_fixed_indices_from_mask(self.val_masks[image_id:image_id+1])
        
        self.name_addon = ""
        input_img_name= f"input_{self.category}_img{image_id}"
        save_path = f"{self.results_dir}/{input_img_name}.png"
        print(f"Saving input image @ {save_path}")
        save_image(self.X_val_np[image_id], save_path)
    
    # Initialize with evolution for multiple samples
    def initCertifAiWithMultipleSamples(self, samples_ids:list[int]):
        self.isCertifAiInitialized = True
        self.datamodule.setup()
        
        images =[self.X_val_np[i:i+1] for i in samples_ids if i < len(self.X_val_np)]
        self.cert = CERTIFAI(numpy_dataset=np.array(images))
        self.cert.set_distance("SSIM")
        masks =[self.val_masks[i:i+1] for i in samples_ids if i < len(self.val_masks)]
        self.maskIndicies = get_fixed_indices_from_mask(masks[0])
        
        self.name_addon = "_ms"
        input_img_name= f"input_{self.category}_ms"
        save_path = f"{self.results_dir}/{input_img_name}.png"
        print(f"Saving input image @ {save_path}")
        combined_image = np.array(np.concatenate(images, axis=2)).reshape(3,self.width*len(samples_ids),self.height)
        save_image(combined_image, save_path)
        
    def fit(self,population = 10, generations = 2):
        if not self.isCertifAiInitialized:
            raise Exception("Please call initCertifAi() before using fit")
        self.model.to("cuda")
        
        self.cert.fit(self.model, generations=generations, gen_retain=min(1,int(population*0.8)), classification=False,verbose=True, constrained=True, fixed = self.maskIndicies, max_population = population)
                
        print("\n - - - - - - - - - - - - -\n   Finished evolution!!!!\n - - - - - - - - - - - - -")
        img_name= f"output_{self.category}_pop{population}_gen{generations}{self.name_addon}"
        save_path = f"{self.results_dir}/{img_name}.png"
        print(f"Saving output image @ {save_path}")
        
        resultImage = np.array(self.cert.results[1]).reshape((3,self.width,self.height))
        save_image(resultImage, save_path)
        
    #Starting population will still still use anomalous sample, but in distance function it will compare to some image classified as normal
    def fitWithTrainImageAsSample(self,population = 10, generations = 2, sampleImageId = 0):
        if not self.isCertifAiInitialized:
            raise Exception("Please call initCertifAi() before using fit")
        self.model.to("cuda")
        
        self.cert.fit_with_good_reference(self.model,self.X_train_np[sampleImageId:sampleImageId+1] , generations=generations, gen_retain=min(1,int(population*0.8)), classification=False,verbose=True, constrained=True, fixed = self.maskIndicies, max_population = population)
                
        print("\n - - - - - - - - - - - - -\n   Finished evolution!!!!\n - - - - - - - - - - - - -")
        self.name_addon = "_with_normal_sample"
        img_name= f"output_{self.category}_pop{population}_gen{generations}{self.name_addon}"
        save_path = f"{self.results_dir}/{img_name}.png"
        print(f"Saving output image @ {save_path}")
        
        resultImage = np.array(self.cert.results[1]).reshape((3,self.width,self.height))
        save_image(resultImage, save_path)





