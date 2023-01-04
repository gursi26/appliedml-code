from model import VGGEncoder, VGGDecoder, AdaIN_realign
from loss import AdaINLoss
from dataset import AdaINDataset

from torch import optim
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class TrainAdaIN:

    def __init__(self, epochs, style_weight, lr, batch_size, content_path, style_path,
                test_content, test_style, dev, enc_weight_path=None, dec_weight_path=None, show_test_output=False) -> None:

        self.dev = dev
        self.style_weight = style_weight
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.show_test_output = show_test_output

        self.load_weights(enc_weight_path, dec_weight_path)

        self.adain_loss = AdaINLoss(self.enc, self.style_weight)

        self.dataset = AdaINDataset(content_path, style_path, batch_size)
        self.test_images_init(test_content, test_style)


    def load_weights(self, enc_weight_path, dec_weight_path):
        if enc_weight_path:
            self.enc = VGGEncoder(weight_path=enc_weight_path).to(self.dev)
        else:
            self.enc = VGGEncoder().to(self.dev)

        if dec_weight_path:
            self.dec = VGGDecoder(self.enc, weight_path=dec_weight_path).to(self.dev)
        else:
            self.dec = VGGDecoder(encoder=self.enc).to(self.dev)


    def test_images_init(self, content_path, style_path):
        self.content_test = Image.open(content_path)
        self.style_test = Image.open(style_path).resize(self.content_test.size)

        T = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])
        self.to_pil = transforms.ToPILImage()

        self.content_test, self.style_test = T(self.content_test), T(self.style_test)
        self.content_test, self.style_test = self.content_test.unsqueeze(0), self.style_test.unsqueeze(0)
        self.content_test, self.style_test = self.content_test.to(self.dev), self.style_test.to(self.dev)

        self.content_test = self.enc(self.content_test)
        self.style_test = self.enc(self.style_test)
        self.realigned_content_test = AdaIN_realign(self.style_test[-1], self.content_test[-1])


    def save_checkpoint(self, save_path = "weights/dec.pth"):
        torch.save(self.dec.state_dict(), save_path)
        

    def show_output(self):
        test_pred_img = self.dec(self.realigned_content_test)[0].clip(0, 1)
        test_pred_img = self.to_pil(test_pred_img)
        test_pred_img.save("output.jpg")


    def train(self):
        opt = optim.Adam(self.dec.parameters(), lr=self.lr)

        for e in range(1, self.epochs + 1):
            loop = tqdm(
                enumerate(zip(self.dataset.content_loader, self.dataset.style_loader)), 
                total=len(self.dataset.style_loader), 
                leave=False, 
                position=0
            )
            loop.set_description(f"Epoch - {e} | ")
            for i, ((content, _), (style, _)) in loop:
                content, style = content.to(self.dev), style.to(self.dev)
                opt.zero_grad()
                self.dec.train()

                content_features = self.enc(content)
                style_features = self.enc(style)
                realigned_content = AdaIN_realign(style_features[-1], content_features[-1])
                pred_img = self.dec(realigned_content).clip(0,1)

                loss = self.adain_loss.calculate_loss(style_features, pred_img, realigned_content)
                loss.backward()
                opt.step()

                loop.set_postfix(loss=loss.item())

                if i % 10 == 0:
                    self.save_checkpoint()
                    if self.show_test_output:
                        self.dec.eval()
                        self.show_output()
