import os
import argparse
from diffusion.diffusion import *
from diffusion.unet import *
from utils.lpips import LPIPS
from utils.utils import test_diffuser
from utils.utils import caption_mode
from transformers import AutoTokenizer
from scipy.linalg import sqrtm
from PIL import Image
from torchvision import models
import warnings
import pandas as pd
# from utils.configs import test_amount
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def lpips_evaluation(args, test_samples_path, real_test_images):
    lpips = LPIPS(args.device)
    lpips.eval()
    lpips.to(args.device)

    real_test_images = real_test_images[:len(os.listdir(test_samples_path))]
    lpips_res = []
    for i in tqdm(range(len((os.listdir(test_samples_path)))), desc="Computing LPIPS"):
        real_image = Image.open(real_test_images[i]+".jpg")
        real_image = torchvision.transforms.ToTensor()(real_image).unsqueeze(0).to(args.device)
        real_image = torch.nn.functional.interpolate(real_image, size=(64, 64), mode='bilinear', align_corners=False)

        fake_image = Image.open(os.path.join(test_samples_path, f"{i+1}.jpg"))
        fake_image = torchvision.transforms.ToTensor()(fake_image).unsqueeze(0).to(args.device)

        lpips_res.append(lpips(real_image, fake_image).item())
    
    return lpips_res

def extract_features(images, device):
    model = models.inception_v3(pretrained=True)
    model = torch.jit.script(model)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        features = model(images)
    return features

def compute_fid(real_features, fake_features):
    real_features = real_features.logits.cpu().numpy()
    fake_features = fake_features.logits.cpu().numpy()

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features)
    
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return fid

def fid_evaluation(args, test_samples_path, real_test_images):
    real_test_images = real_test_images[:len(os.listdir(test_samples_path))]
    fid_res = []

    batch_size = len(os.listdir(test_samples_path))
    # len((os.listdir(test_samples_path))
    for i in tqdm(range(0, len(os.listdir(test_samples_path)), batch_size), desc="Computing FID"):
        real_opened_images = []
        fake_opened_images = []
        
        for j in range(i, i+batch_size):
            try:
                real_image = Image.open(real_test_images[j]+".jpg")
                real_image = torchvision.transforms.ToTensor()(real_image).unsqueeze(0).to(args.device)
                real_image = torch.nn.functional.interpolate(real_image, size=(80, 80), mode='bilinear', align_corners=False)
                real_image = real_image.mean(1, keepdim=True)
                real_image = real_image.repeat(1, 3, 1, 1)

                fake_image = Image.open(os.path.join(test_samples_path, f"{j+1}.jpg"))
                fake_image = torchvision.transforms.ToTensor()(fake_image).unsqueeze(0).to(args.device)
                fake_image = torch.nn.functional.interpolate(fake_image, size=(80, 80), mode='bilinear', align_corners=False)
                fake_image = fake_image.mean(1, keepdim=True)
                fake_image = fake_image.repeat(1, 3, 1, 1)

                real_opened_images.append(real_image)
                fake_opened_images.append(fake_image)
            except:
                continue

        real_opened_images = torch.cat(real_opened_images, dim=0)
        fake_opened_images = torch.cat(fake_opened_images, dim=0)

        real_features = extract_features(real_opened_images, args.device)
        fake_features = extract_features(fake_opened_images, args.device)


        fid_res.append(compute_fid(real_features, fake_features))

    return fid_res

def main(args):
    results_path = args.path
    models_path = os.path.join(results_path, "model")
    test_samples_path = os.path.join(results_path, "test_samples")

    if not os.path.exists(test_samples_path):
        if not os.path.exists(test_samples_path):
            print("Creating test samples directory...\n")
            os.mkdir(test_samples_path)

        frozen = "Frozen" in results_path

        model = None
        model_config = torch.load(os.path.join(models_path, os.listdir(models_path)[0]), map_location=args.device)
        if "VGG" in results_path:
            model = VGG16_Unet(device=args.device, image_size=model_config['image_size'], use_clip = model_config['use_clip'], freeze=frozen).to(args.device)
        else:
            model = UNet(device=args.device, image_size=model_config['image_size'], use_clip = model_config['use_clip']).to(args.device)
        model.load_state_dict(model_config["model_state_dict"])
        model.eval()

        diffusion = Diffusion(img_size=model_config['image_size'], device=args.device, scheduler_type="linear")
        text_tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')

        cm = None
        if "caption_mode" in model_config.keys():
            cm = model_config["caption_mode"]
        else:
            cm = caption_mode["KEYWORDS"] if model_config['use_keywords'] else caption_mode["CAPTION"]
            cm = caption_mode["CAPTION_CUI"] if "Cui" in results_path else cm

        print("Caption Mode:", cm, "\n")

        print("Generating samples...\n")
        test_diffuser(
            model, 
            diffusion, 
            text_tokenizer, 
            test_samples_path, 
            image_size = model_config['image_size'], 
            k = model_config['k'], 
            use_clip=model_config['use_clip'], 
            cm=cm,
            device = args.device,
            skip_amount = len(os.listdir(test_samples_path))
        )


    test_fid = fid_evaluation(args, test_samples_path, args.test_images)
    print(f"Mean FID: {np.mean(test_fid)}\n")

    with open(os.path.join(results_path, "evaluation_results.txt"), 'w') as f:
        f.write(f"Mean FID: {np.mean(test_fid)}\n")
    
if __name__ == '__main__':
    results = sorted(os.listdir("./results"))
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', dest='device', default='cuda')

    
    for i, res in enumerate(results):
        if "Net" not in res:
            continue

        print(f"Evaluating {res}")
        k = 20 if "k20" in res else 10
    
        test_images_df = pd.read_csv(f"./data/rocov2/processed/test_top{k}_kcf.csv")
        test_images = [os.path.join("./data/rocov2/test_images", p) for p in test_images_df["ID"].values]
        
        args = parser.parse_args()
        args.path = os.path.join("./results", res)
        args.test_images = test_images
        
        main(args)

        print()