import json
import torch
import clip_classifier
import clip
from PIL import Image 


class OoCICIdentification:
    def __init__(self, clip_model="ViT-B/32"):
        self.device = "mps" if torch.backends.mps.is_built() else "cpu"
        self.model_settings = {'pdrop':0.5}
        self.base_clip, self.preprocess = clip.load(clip_model, self.device)
        self.classifier_clip = clip_classifier.ClipClassifier(self.model_settings,self.base_clip).to(self.device)

    def load_model(self, clip_classifier_path):
        checkpoint = torch.load(clip_classifier_path, map_location=self.device)
        self.classifier_clip.load_state_dict(checkpoint['state_dict'])
        clip.model.convert_weights(self.classifier_clip)
        
    def process_img_caption(self, image_path, caption):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            pil_img = img.convert('RGB')
            
        transform_img = self.preprocess(pil_img)
        caption_tokenized = clip.tokenize(caption)
    
        return transform_img, caption_tokenized

    def predictions(self, input_file, images_folder):
        
        correctly_identified = 0
        incorrectly_identified = 0
        num_samples = 0
        with open(input_file, 'r') as json_lines:
            # save the predictions along with imagea and caption
            with open("output_file.txt", 'w') as file:
                for line in json_lines:

                    sample = json.loads(line) 

                    img_path = images_folder + sample['img_local_path'] 
                    caption = sample['caption1']
                    label = sample['context_label']
                    splitted_caption = caption.split(" ")
                    caption_len = len(splitted_caption)
                    # print(caption_len)
                    if caption_len > 55:
                        caption = " ".join(splitted_caption[:55])

                    processed_image, tokenized_caption = self.process_img_caption(img_path, caption)

                    img = processed_image.unsqueeze(0).to(self.device)
                    cap = tokenized_caption.to(self.device)

                    output = self.classifier_clip(img,cap)

                    pred = 1.0*(torch.sigmoid(output) >= 0.5).item()

                    if pred == label:
                        correctly_identified += 1
                    else:
                        incorrectly_identified += 1
                    num_samples += 1

                    
                    #     file.write(sample['img_local_path'] + caption + (pred.item())*1.0)
                    file.write(f"{sample['img_local_path']}\t{caption}\t{pred}\n")

        return (correctly_identified/num_samples)*100.0, (incorrectly_identified/num_samples)*100.0
    
def main():

    input_data = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/Ookpik_dataset/test_data.json'
    input_images = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/Ookpik_dataset/'
    model_path = '/Users/komalkrishnamogilipalepu/Downloads/OoC-multi-modal-fc-main/exp/best_model_acc.pth.tar'
    
    IC_identification = OoCICIdentification()
    IC_identification.load_model(model_path)
    
    correctly_classified, incorrectly_classified = IC_identification.predictions(input_data, input_images)

    print("Percentage of correctly classified", correctly_classified)
    print("Percentage of incorrectly classified", incorrectly_classified)

if __name__ == "__main__":
    main()

            





