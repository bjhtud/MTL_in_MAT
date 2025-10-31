import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, dim, theta= 8):
        super(Autoencoder, self).__init__()
        self.dim = dim
        self.theta = theta
        self.drop_out = nn.Dropout(p=0.5)

        self.encoder = nn.Sequential(
            nn.Linear(dim+theta*0, dim+theta*1),
            nn.Tanh(),
            nn.Linear(dim+theta*1, dim+theta*2),
            nn.Tanh(),
            nn.Linear(dim+theta*2, dim+theta*3),
            nn.Tanh(),
            nn.Linear(dim+theta*3, dim+theta*4),
            nn.Tanh(),
            nn.Linear(dim+theta*4, dim+theta*5)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim+theta*5, dim+theta*4),
            nn.Tanh(),
            nn.Linear(dim+theta*4, dim+theta*3),
            nn.Tanh(),
            nn.Linear(dim+theta*3, dim+theta*2),
            nn.Tanh(),
            nn.Linear(dim+theta*2, dim+theta*1),
            nn.Tanh(),
            nn.Linear(dim+theta*1, dim+theta*0)
        )

        self.scaler = None
        self.device = None

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out

    def fit(
            self,
            train_data,
            num_epochs = 100,
            batch_size = 1,
            ):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.to(self.device)

        train_data = train_data.fillna(0)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        train_data = self.scaler.transform(train_data)

        train_data = torch.from_numpy(train_data).float()
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        loss_fn = nn.MSELoss()

        optimizer = optim.SGD(self.parameters(), momentum=0.99, lr=0.01, nesterov=True)

        for epoch in range(num_epochs):
            for batch_data in train_loader:
                batch_data = batch_data.to(self.device)

                reconst = self(batch_data)
                loss = loss_fn(reconst, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def transform(self, data):
        self.eval()

        missing_mask = data.isna()

        data_filled = data.fillna(0)
        data_scaled = self.scaler.transform(data_filled)
        data_tensor = torch.from_numpy(data_scaled).float().to(self.device)
        with torch.no_grad():
            reconstructed = self(data_tensor)

        reconstructed = reconstructed.cpu().numpy()
        reconstructed = self.scaler.inverse_transform(reconstructed)

        reconstructed_df = pd.DataFrame(reconstructed, columns=data.columns, index=data.index)

        imputed = data.copy()
        imputed[missing_mask] = reconstructed_df[missing_mask]

        return imputed
