from mvTek import CertifaiMvTekWrapper
import os

if __name__ == "__main__":

    winPath = "f:/datasets/mvtec/"
    wslPath = "/mnt/f/datasets/mvtec/"
    data_path = wslPath
    if os.name == "nt":
        data_path = winPath

    certifAi_bottle = CertifaiMvTekWrapper(dataset_dir = data_path, category="bottle", results_dir="./results/")
    
    certifAi_bottle.initCertifAi(20)
    
    certifAi_bottle.fit(population=5, generations=2)
    certifAi_bottle.fitWithTrainImageAsSample(population=5, generations=2, sampleImageId = 0)
    
