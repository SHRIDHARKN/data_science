import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from rich.console import Console

console = Console()

def show_blue_msg(msg):
    console.print(msg, style="bold #003BFF")

def show_reg_msg(msg):
    console.print(msg, style="bold #FF003B")

def show_green_msg(msg):
    console.print(msg, style="bold #2AB051")

def show_orange_msg(msg):
    console.print(msg, style="bold #FF8A00")

def show_pink_msg(msg):
    console.print(msg, style="bold #FE00FF")
    
def get_train_test_val_img_dataloaders(data_dir, img_height, img_width, batch_size, help=False):

    if help:
        return """
        train_loader, test_loader, val_loader = get_train_test_val_img_dataloaders(data_dir, img_height, img_width,batch_size)
        """
    show_blue_msg(f"data_dir path: {data_dir}")
    show_blue_msg(f"found subfolds: {os.listdir(data_dir)}")
    show_reg_msg(f"img_height, img_weight: {img_height}, {img_width}")
    show_reg_msg(f"batch_size: {batch_size}")
        
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    show_blue_msg(f"image loader is ready")
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    show_blue_msg(f"train_loader is ready")
    show_blue_msg(f"val_loader is ready")
    show_blue_msg(f"test_loader is ready")
    
    return train_loader, test_loader, val_loader
