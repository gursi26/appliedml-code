from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


class AdaINDataset:

    def __init__(self, content_path, style_path, batch_size) -> None:

        self.T = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop((256, 256), padding=(20, 20)),
            transforms.ToTensor(),
        ])

        self.content_folder = ImageFolder(content_path, transform=self.T)
        self.style_folder = ImageFolder(style_path, transform=self.T)

        self.content_loader = DataLoader(self.content_folder, batch_size, shuffle=True)
        self.style_loader = DataLoader(self.style_folder, batch_size, shuffle=True)


def test_dataset():
    content_path = "/Users/gursi/desktop/content"
    style_path = "/Users/gursi/desktop/style"
    dataset = AdaINDataset(content_path, style_path, 8)
    print(dataset.content_folder[0])
    T = transforms.ToPILImage()
    (content, _), (style, _) = next(zip(dataset.content_loader, dataset.style_loader))
    
    fig, ax = plt.subplots(nrows=2, ncols=8)
    for i in range(content.shape[0]):
        ax[0][i].imshow(T(content[i]))
        ax[0][i].axis(False)
        ax[1][i].imshow(T(style[i]))
        ax[1][i].axis(False)

    plt.show()


if __name__ == "__main__":
    test_dataset()