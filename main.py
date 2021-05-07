import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST(
    root='./datasets',
    train=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]),
    download=True)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=2,
    shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AE(nn.Module):
    def __init__(self, device):
        super(AE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, 20),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode the images
        encs = self.encoder(x)

        # Get the components from the encodes
        means = encs[:,:10]
        sds = encs[:,10:]

        # Sample
        samples = (sds * torch.randn_like(means)) + means
        
        # Decode the images
        imgs = self.decoder(samples)

        # Return the images in the proper format
        return means, sds, imgs.view(-1, 1, 28, 28)

if __name__ == '__main__':
    # Number of epochs
    EPOCHS = 2

    # Model load
    LOAD = 'model.pth'
    # LOAD = False

    # Create the model and send it to the GPU
    model = AE(device)
    model = model.to(device)

    # Create the criterons
    enc_crit = nn.MSELoss()
    img_crit = nn.MSELoss()

    # Create the optimizer
    adam = optim.Adam(model.parameters(), lr=0.0005)

    if LOAD is False:
        for epoch in range(EPOCHS):
            for i, data in enumerate(loader):
                # Unpack the data
                digits = data[1].to(device)
                data = data[0].to(device)

                # Zero out the gradients
                adam.zero_grad()

                # Create the images
                means, sds, images = model(data)

                # Create the labels
                idx = torch.arange(digits.shape[0]).unsqueeze(-1)
                # labels = torch.randn_like(means, device=device)
                # labels[idx.reshape(-1), digits.view(-1)] = torch.randn(digits.shape[0], device=device) + 1
                # labels[labels < 0] = 0
                # labels[labels > 1] = 1
                labels = torch.zeros_like(means, device=device)
                labels[idx.reshape(-1), digits.view(-1)] += 10
                
                # Compute the loss & backpropagate
                enc_loss = enc_crit(means, labels)
                img_loss = img_crit(images, data)


                # Helpful print
                if i%32 == 0:
                    print(f'Loss ({epoch}, {i}) {enc_loss.item()} {img_loss.item()}')

                # Backpropagate
                enc_loss.backward(retain_graph=True)
                img_loss.backward(retain_graph=True)
                adam.step()

        # Save the model
        torch.save(model.state_dict(), 'model.pth')

    else:
        # Reset the device
        device = torch.device('cpu')

        # Load the model
        model = AE(device)
        model.load_state_dict(torch.load(LOAD))
    
    enc = torch.randn(9, 10, device=device).float()
    enc[0,3] += 10
    enc[0,7] += 0

    enc[1,3] += 9
    enc[1,7] += 1

    enc[2,3] += 8
    enc[2,7] += 2

    enc[3,3] += 7
    enc[3,7] += 3

    enc[4,3] += 5
    enc[4,7] += 5

    enc[5,3] += 3
    enc[5,7] += 7

    enc[6,3] += 2
    enc[6,7] += 8

    enc[7,3] += 1
    enc[7,7] += 9

    enc[8,3] += 0
    enc[8,7] += 10
    enc = enc.unsqueeze(0)
    image = model.decoder(enc)
    image = image.reshape(9, 28, 28, 1).cpu().detach().numpy()

    fig, axis = plt.subplots(3, 3)
    axis[0,0].imshow(image[0])
    axis[0,1].imshow(image[1])
    axis[0,2].imshow(image[2])
    axis[1,0].imshow(image[3])    
    axis[1,1].imshow(image[4])
    axis[1,2].imshow(image[5])
    axis[2,0].imshow(image[6])
    axis[2,1].imshow(image[7])
    axis[2,2].imshow(image[8])
    plt.show()