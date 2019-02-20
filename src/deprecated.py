
### TOOLS FOR CONTRACTION TRAINERS

def jacobian(inputs, outputs):
    return torch.stack([torch.autograd.grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0]
                        for i in range(outputs.size(1))], dim=-1)


class contractive_ae():
    def __init__(self, optim = 'adam', use_cuda = False, nclass = 10, l_contr = 1,
        l_anchor = 0, eps_c = 0.2, anchor = None, denoising = False,  *args, **kwargs):
        super(contractive_ae, self).__init__()
        self.net = FFCAE()
        self.log_interval = kwargs['log_interval']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.nclass = nclass
        self.batch_size = kwargs['batch_size']
        self.lambda_contraction = l_contr
        self.lambda_anchor      = l_anchor
        self.eps_contraction = eps_c
        self.denoising       = denoising
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-5)
        if anchor is not None:
            self.x_anchor = anchor
            self.use_anchor_loss = True
        else:
            self.use_anchor_loss = False

    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net, open(path, 'wb'))
        print('Saved!')

    def train(self,  train_loader, test_loader, epochs = 2):
        # Train classifier
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.test(test_loader, epoch)

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        train_loss = 0
        if self.denoising:
            noise = torch.rand(self.batch_size,1,28,28)
        for idx, (data,_) in enumerate(train_loader):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), data.to(self.device).detach()
            if self.denoising:
                if data.shape[0] != noise.shape[0]: # last batch
                    noise = torch.rand(data.shape[0],1,28,28)
                data = torch.mul(data+0.25, 0.1 * noise)
            data.requires_grad = True

            # 1. Reconstruction loss
            recons_x = self.net(data)
            recons_loss = F.mse_loss(recons_x, target) #+ lam*grad_penalty
            losses = [recons_loss]

            # 2. Contraction Loss - via Jacobian
            dF = jacobian(data, recons_x)
            #grad_penalty = (dF).norm(2) #.pow(2) # ABSOLUTE
            contraction_loss = (dF.norm(2) - (1 - self.eps_contraction)).clamp(min = 0) # Only penalize above eps
            losses.append(self.lambda_contraction*contraction_loss)

            # 3. Optional: Anchor loss to guide fixed point
            if self.use_anchor_loss:
                anchor_loss = F.mse_loss(self.x_anchor,self.net(self.x_anchor))
                losses.append(self.lambda_anchor*anchor_loss)
            else:
                anchor_loss = torch.zeros(1)

            sum(losses).backward()
            self.optimizer.step()

            train_loss += sum(losses).item()

            if idx % self.log_interval == 0:
                print('Train epoch: {} [{}/{}({:.0f}%)]\t '
                 'Rec. Loss: {:.2e}\t Contr. Loss: {:.2e}\t FP Loss: {:.2e}'.format(
                  epoch, idx*len(data), len(train_loader.dataset),
                  100*idx/len(train_loader),
                  recons_loss.item()/len(data), contraction_loss.item()/len(data),
                  anchor_loss.item()/len(data)))

        print('====> Epoch: {} Average training loss: {:.8f}'.format(
             epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader, epoch):
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data, target = data.to(self.device), data.to(self.device)
                output = self.net(data)
                batch_loss = F.mse_loss(output, target, size_average=True).item()
                #batch_loss += (dF.norm(2) - (1 - self.eps_contraction)).clamp(min = 0)
                test_loss +=  batch_loss

                #TODO: add contractive loss here too
        self.net.samples_write(data,epoch)
        plt.show()
        test_loss /= len(test_loader.dataset)
        print('====> Epoch: {} Average training loss: {:.8f}\n\n'.format(epoch, test_loss))

        if self.use_anchor_loss:
            x_star_rec =self.net(self.x_anchor)
            print((x_star_rec - self.x_anchor).norm().detach().numpy())
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(self.x_anchor.squeeze())
            ax[1].imshow(x_star_rec.detach().squeeze())
            plt.show()




class class_contractive_ae():
    def __init__(self, optim = 'adam', use_cuda = False, nclass = 10, lambd = 1,
        contr_steps = 1, eps_c = 0.2, eps_e = 0.5, *args, **kwargs):
        super(class_contractive_ae, self).__init__()
        self.net = FFCAE()
        self.log_interval = kwargs['log_interval']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.nclass = nclass
        self.batch_size = kwargs['batch_size']
        self.lambda_contraction = lambd
        self.eps_contraction = eps_c
        self.eps_expansion   = eps_e
        self.steps_contraction = contr_steps
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-5)
        self.class_loaders = {}

    def __call__(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net, open(path, 'wb'))
        print('Saved!')

    def train(self, pred_y, train_loader, test_loader, epochs = 2):
        self.init_class_loaders(train_loader.dataset, pred_y)
        # Train classifier
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            #self.test(test_loader)

    def init_class_loaders(self, dataset, pred_y):
        for klass in range(self.nclass):
            weights = (pred_y == klass)
            sampler = WeightedRandomSampler(weights, len(weights), replacement = True)
            self.class_loaders[klass] = iter(torch.utils.data.DataLoader(dataset=dataset,
                           batch_size=self.batch_size,drop_last = True,
                           sampler = sampler))#, num_workers=args.workers, pin_memory=True)
        #return class_loaders
    def reset_class_loader(self, k):
        sampler = self.class_loaders[k].batch_sampler.sampler
        loader  = torch.utils.data.DataLoader(dataset=self.class_loaders[k].dataset,
                               batch_size=self.batch_size,drop_last = True,
                               sampler = sampler)
        self.class_loaders[k] = iter(loader)#, num_worker

    def contraction_loss(self, XC1, XC2, XC1_r, XC2_r):
        """
            *_r are the outputs (reconstructed)
        """

        # Split into two
        b = self.batch_size
        n = b/2
        XC1_A, XC1_B = torch.split(XC1.view(b,-1), [n, n], dim = 0)
        XC2_A, XC2_B = torch.split(XC2.view(b,-1), [n, n], dim = 0)
        XC1_A_r, XC1_B_r = torch.split(XC1_r.view(b,-1), [n, n], dim = 0)
        XC2_A_r, XC2_B_r = torch.split(XC2_r.view(b,-1), [n, n], dim = 0)

        # Compute Lipschitz Ratios
        # lip_C1  = (XC1_A_r - XC1_B_r).norm()/(XC1_A - XC1_B).norm()
        # lip_C2  = (XC2_A_r - XC2_B_r).norm()/(XC2_A - XC2_B).norm()
        # lip_12  = (XC1_A_r - XC2_A_r).norm()/(XC1_A - XC2_A).norm()
        # lip_21  = (XC1_B_r - XC2_B_r).norm()/(XC1_B - XC2_B).norm()
        # Each of this is (n x 1)
        lip_C1  = (XC1_A_r - XC1_B_r).norm(dim=1)/(XC1_A - XC1_B).norm(dim=1)
        lip_C2  = (XC2_A_r - XC2_B_r).norm(dim=1)/(XC2_A - XC2_B).norm(dim=1)
        lip_12  = (XC1_A_r - XC2_A_r).norm(dim=1)/(XC1_A - XC2_A).norm(dim=1)
        lip_21  = (XC1_B_r - XC2_B_r).norm(dim=1)/(XC1_B - XC2_B).norm(dim=1)


        # For same class, we want contraction: lip <= (1 - eps_c)
        contraction_loss = (lip_C1 -  (1 - self.eps_contraction)).clamp(min = 0) + \
                           (lip_C2 -  (1 - self.eps_contraction)).clamp(min = 0)
        # Across classes, we want expansion:   lip >= (1 + eps_e)
        expansion_loss =  -(lip_12 - (1 + self.eps_expansion)).clamp(max = 0) + \
                          -(lip_21 - (1 + self.eps_expansion)).clamp(max = 0)
        # Total loss is sum of the two
        return contraction_loss.mean() + expansion_loss.mean()

    def _draw_from_class(self, k):
        try:
            x, _ = next(self.class_loaders[k])
            assert x.shape[0] == self.batch_size#class_loaders[klass].batch_sampler.batch_size
        except:
            print('Reset iterator for class: ', k)
            self.reset_class_loader(k)
            # class_loaders[klass] = iter(torch.utils.data.DataLoader(dataset=dataset,
            #        batch_size=args.batch_size,
            #        sampler = sampler))#, num_workers=args.workers, pin_memory=True)
            x, _ = next(self.class_loaders[k])
        return x

    def train_epoch(self, train_loader, epoch):
        self.net.train()
        train_loss = 0
        for idx, (data,_) in enumerate(train_loader):
            data, target = data.to(self.device), data.to(self.device)
            self.optimizer.zero_grad()

            recons_x = self.net(data)
            recon_loss = F.mse_loss(recons_x, data.detach()) #+ lam*grad_penalty
            losses = [recon_loss]

            # Randomly choose a pair of distinct classes, map them
            for i in range(self.steps_contraction):
                c1, c2 = np.random.choice(self.nclass, 2, replace=False)
                XC1 = self._draw_from_class(c1)
                XC2 = self._draw_from_class(c2)
                XC1_rec = self.net(XC1)
                XC2_rec = self.net(XC2)
                contraction_loss = self.contraction_loss(XC1, XC2, XC1_rec, XC2_rec)
                losses.append(self.lambda_contraction*contraction_loss)

            sum(losses).backward()
            self.optimizer.step()
            train_loss += sum(losses).item()

            if idx % self.log_interval == 0:
                print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.8f}\t '
                 'Rec. Loss: {:.8f}\t Contr. Loss: {:.8f}'.format(
                  epoch, idx*len(data), len(train_loader.dataset),
                  100*idx/len(train_loader),sum(losses).item()/len(data),
                  recon_loss.item()/len(data), contraction_loss.item()/len(data)))

        print('====> Epoch: {} Average loss: {:.8f}'.format(
             epoch, train_loss / len(train_loader.dataset)))
        self.net.samples_write(data,epoch)

    def test(self, test_loader):
        self.net.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data, target = data.to(self.device), data.to(self.device)
                output = self.net(data)
                test_loss += F.mse_loss(output, target, size_average=True).item() # sum up batch loss

                #TODO: add contractive loss here too

        test_loss /= len(test_loader.dataset)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data[0].squeeze())
        ax[1].imshow(output[0].squeeze())
        plt.show()

        print('\nTest set: Average loss: {:.8f}\n'.format(test_loss))
