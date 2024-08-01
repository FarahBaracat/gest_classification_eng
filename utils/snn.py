from torch.utils.data import DataLoader, Subset


def prepare_data_loaders(dataset, train_ind, test_ind, monitor_indices, batch_size):
    train_sub = Subset(dataset, train_ind)
    test_sub = Subset(dataset, test_ind)
    monitor_sub = Subset(dataset, monitor_indices)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=True)
    monitor_loader = DataLoader(monitor_sub, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, monitor_loader
