import torch

import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
from PIL import Image
from easyfsl.data_tools import TaskSampler
from easyfsl.utils import plot_images, sliding_average


image_size = 224
 

train_tranform = transforms.Compose(
        [
            # transforms.Grayscale(num_output_channels=2),
            # transforms.RandomResizedCrop(image_size),
            transforms.Resize(image_size),
            # transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
            

test_tranform =transform=transforms.Compose(
        [
           
            # transforms.Grayscale(num_output_channels=2),
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
   
    
train_set = torchvision.datasets.ImageFolder(root="few_shot_data - Copy//train/", transform = train_tranform)
# test_set = torchvision.datasets.ImageFolder(root= "few_shot_data/val/", transform = test_tranform)

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
# print(convolutional_network)

model = PrototypicalNetworks(convolutional_network)
model.load_state_dict(torch.load('8_shot.pth',map_location={'cuda:0': 'cpu'}))

N_TRAINING_EPISODES = 2000
N_VALIDATION_TASKS = 100

train_set.labels = [instance[1] for instance in train_set.imgs]
train_sampler = TaskSampler(
    train_set, n_way=2, n_shot=8, n_query=2, n_tasks=N_TRAINING_EPISODES
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)


(
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
) = next(iter(train_loader))


def transform_image(img):
    import io
    from prep_image_module import prep_img
    import cv2
    import matplotlib.pyplot as plt
    tranform = transforms.Compose(
        [
           
            # transforms.Grayscale(num_output_channels=2),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]
    )
    # image  = Image.open(path)
    image = prep_img(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    # pil_image.show()
    # image = Image.open(io.BytesIO(image_bytes))
    image_tensor = tranform(pil_image).unsqueeze(0)
    
    return image_tensor


def prediction(image_tensor):
    example_scores = model(  example_support_images,  example_support_labels,image_tensor).detach()
    _, example_predicted_labels = torch.max(example_scores.data, 1)
    # return example_predicted_labels
    return "Normal" if example_predicted_labels.numpy()[0]==1 else "Cataract"
