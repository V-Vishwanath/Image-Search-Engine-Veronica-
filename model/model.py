import torch
import pickle
import argparse
import numpy as np
from torch import nn  
from PIL import Image
from torchvision import transforms 

from objects import Encoder, Decoder, Vocabulary

class ImageToSentence(nn.Module):
    def __init__(self, model_data_path):
        super(ImageToSentence, self).__init__()

        # init paths
        VOCAB_FILE = f'{model_data_path}/vocab.pkl'
        ENCODER_WEIGHTS = torch.load(f'{model_data_path}/encoder.pt')
        DECODER_WEIGHTS = torch.load(f'{model_data_path}/decoder.pt')

        # Load vocab data
        with open(VOCAB_FILE, 'rb') as f:
            self.vocab = pickle.load(f)

        # Create Encoder and Decoder
        self.encoder = Encoder(256)
        self.decoder = Decoder(256, 512, len(self.vocab), 2)

        self.encoder.load_state_dict(ENCODER_WEIGHTS)
        print('[INFO]  Loaded Encoder successfully!')
        self.decoder.load_state_dict(DECODER_WEIGHTS)
        print('[INFO]  Loaded Decoder successfully!')

        self.encoder.eval()
        self.decoder.eval()


    def get_image_tensor(self, image):
        get_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return get_tensor(image).unsqueeze(0)

    
    def forward(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize([224, 224], Image.LANCZOS)

        tensor = self.get_image_tensor(img)

        features = self.encoder(tensor)
        _ids = self.decoder(features)[0].numpy()

        caption = []
        for _id in _ids:
            word = self.vocab.idx2word[_id]
            caption.append(word)
            if word == '<end>': break 

        return ' '.join(caption[1:-1])