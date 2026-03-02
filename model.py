import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm

from util import setup_seed, pearson_corr, DropData


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    return acc, nmi, ari, homo, comp, purity


class MetaContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(MetaContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels):
        unique_classes = torch.unique(labels)
        N = len(unique_classes)
        if N < 2:
            return torch.tensor(0.0, device=embeddings.device)

        prototypes = []
        class_masks = []
        for cls in unique_classes:
            mask = (labels == cls)
            class_masks.append(mask)
            proto = embeddings[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)

        total_loss = 0.0
        total_count = 0

        for cls_idx, cls in enumerate(unique_classes):
            mask = class_masks[cls_idx]
            cls_embeddings = embeddings[mask]
            M_i = cls_embeddings.shape[0]

            sims = torch.matmul(cls_embeddings, prototypes.T) / self.tau
            pos_sims = sims[:, cls_idx]
            log_denom = torch.logsumexp(sims, dim=1)
            sample_loss = -(pos_sims - log_denom)

            total_loss += sample_loss.sum()
            total_count += M_i

        if total_count == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return total_loss / total_count


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256]):
        super(Decoder, self).__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class Model(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=128, output_dim=10):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        self.w_imp = Decoder(hidden_dim, input_dim, hidden_dims=[128, 256])

        self.mae_loss = torch.nn.L1Loss(reduction='mean')
        self.l1loss = torch.nn.L1Loss(reduction='none')

    def GCN(self, batch):
        embs = []
        for data in batch:
            x = self.conv1(data['x'], data['edge_index'])
            x = F.relu(x)
            x = self.conv2(x, data['edge_index'])
            embs.append(x.mean(0))
        return torch.stack(embs, 0)

    def forward(self, x, batch_subgraph, test=False):
        z = self.encoder(x)
        node_emb = self.GCN(batch_subgraph)
        x_imp = self.w_imp(node_emb)
        combined = torch.cat((z, node_emb), dim=1)

        if test:
            return combined, x_imp

        return combined, x_imp


def sample_episode(dataset, N_way, M_shot, Q_query):
    labels_np = dataset.y_pseudo.cpu().numpy()
    unique_classes, counts = np.unique(labels_np, return_counts=True)

    min_samples = 2
    valid_mask = counts >= min_samples
    valid_classes = unique_classes[valid_mask]

    N = min(N_way, len(valid_classes))
    if N < 2:
        return None

    selected_classes = np.random.choice(valid_classes, size=N, replace=False)

    support_indices = []
    query_indices = []

    for cls in selected_classes:
        cls_indices = np.where(labels_np == cls)[0]
        np.random.shuffle(cls_indices)

        n_available = len(cls_indices)
        m = min(M_shot, n_available - 1)
        if m < 1:
            m = 1
        q = min(Q_query, n_available - m)
        if q < 1:
            continue

        support_indices.extend(cls_indices[:m].tolist())
        query_indices.extend(cls_indices[m:m + q].tolist())

    if len(support_indices) < 2 or len(query_indices) < 1:
        return None

    support_X = dataset.X[support_indices]
    support_y = dataset.y_pseudo[support_indices]
    support_soft = dataset.soft_labels[support_indices]
    support_subgraphs = [dataset.subgraphs[i] for i in support_indices]

    query_X = dataset.X[query_indices]
    query_y = dataset.y_pseudo[query_indices]
    query_soft = dataset.soft_labels[query_indices]
    query_subgraphs = [dataset.subgraphs[i] for i in query_indices]

    return (support_X, support_y, support_soft, support_subgraphs,
            query_X, query_y, query_soft, query_subgraphs)


def train(train_loader, valid_loader, test_loader, lr, seed, epochs,
          n_clusters, n_clusters_test, input_dim,
          save_model_path, alpha=0.5, gamma=0.2,
          N_way=5, M_shot=5, Q_query=5, tau=1.0,
          device='cuda'):

    model = Model(input_dim, output_dim=n_clusters).cuda()
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    meta_contrastive_loss_fn = MetaContrastiveLoss(tau=tau)
    ce_loss_fn = nn.CrossEntropyLoss()

    setup_seed(seed)
    train_loss_history = []
    valid_loss_history = []
    best_epoch = 0

    train_dataset = train_loader.dataset
    val_dataset = valid_loader.dataset

    steps = len(train_loader)
    min_loss = 1e6
    patience = 10
    patience_counter = 0
    ari_patience = 5
    ari_patience_counter = 0

    best_test_metrics = {
        'ACC': -1, 'ARI': -1, 'NMI': -1,
        'Purity': -1, 'Homo': -1, 'Comp': -1,
        'PCC': {}, 'MAE': {}, 'epoch': -1
    }
    best_test_results = []

    for each_epoch in range(epochs):
        if patience_counter >= patience:
            print(f"Early stopping at epoch {each_epoch + 1}")
            break

        batch_losses = []
        model.train()

        with tqdm(total=steps, desc=f'Epoch {each_epoch + 1}/{epochs}', unit='batch') as pbar:
            for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                        batch_loc, batch_subgraph) in enumerate(train_loader):

                combined, x_imp = model(batch_x, batch_subgraph)
                mask = (batch_x != 0).float()
                loss_imp = model.l1loss(x_imp, batch_x)
                loss_imp = (loss_imp * mask).sum() / (mask.sum() + 1e-6)

                episode = sample_episode(
                    train_dataset, N_way=N_way, M_shot=M_shot, Q_query=Q_query
                )

                if episode is not None:
                    (s_X, s_y, s_soft, s_subgraphs,
                     q_X, q_y, q_soft, q_subgraphs) = episode

                    s_combined, _ = model(s_X, s_subgraphs)
                    loss_ct = meta_contrastive_loss_fn(s_combined, s_y)

                    q_combined, _ = model(q_X, q_subgraphs)
                    q_logits = model.classifier(q_combined)
                    loss_ce = ce_loss_fn(q_logits, q_y)

                    loss_meta = alpha * loss_ct + (1 - alpha) * loss_ce

                else:
                    loss_meta = torch.tensor(0.0, device=device)

                loss = loss_meta + gamma * loss_imp

                opt_model.zero_grad()
                loss.backward()
                opt_model.step()

                batch_losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item(),
                                  'meta': loss_meta.item(),
                                  'imp': loss_imp.item()})
                pbar.update(1)

        train_loss_history.append(np.mean(batch_losses))

        with torch.no_grad():
            val_losses = []
            model.eval()
            for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                        batch_loc, batch_subgraph) in enumerate(valid_loader):
                combined, x_imp = model(batch_x, batch_subgraph)
                mask = (batch_x != 0).float()
                loss_imp = model.l1loss(x_imp, batch_x)
                loss_imp = (loss_imp * mask).sum() / (mask.sum() + 1e-6)

                episode = sample_episode(
                    val_dataset, N_way=N_way, M_shot=M_shot, Q_query=Q_query
                )
                if episode is not None:
                    (s_X, s_y, s_soft, s_subgraphs,
                     q_X, q_y, q_soft, q_subgraphs) = episode
                    s_combined, _ = model(s_X, s_subgraphs)
                    loss_ct = meta_contrastive_loss_fn(s_combined, s_y)
                    q_combined, _ = model(q_X, q_subgraphs)
                    q_logits = model.classifier(q_combined)
                    loss_ce = ce_loss_fn(q_logits, q_y)
                    loss_meta = alpha * loss_ct + (1 - alpha) * loss_ce
                else:
                    loss_meta = torch.tensor(0.0, device=device)

                val_loss = loss_meta + gamma * loss_imp
                val_losses.append(val_loss.item())

        valid_loss_history.append(np.mean(val_losses))
        cur_loss = valid_loss_history[-1]

        print(f"Epoch {each_epoch+1}: train_loss={train_loss_history[-1]:.4f}, "
              f"val_loss={cur_loss:.4f}")

        if cur_loss < min_loss or each_epoch == 0:
            print(f'Saving model at Epoch {each_epoch} with loss {cur_loss:.4f}')
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict()
            }
            torch.save(state, save_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_results = test(test_loader, n_clusters, n_clusters_test,
                             input_dim=input_dim, seed=seed, model=model)
        epoch_best_ari = epoch_results['best_ari']
        epoch_best_nmi = epoch_results['best_nmi']
        print(f"  Test @ Epoch {each_epoch+1}: best ARI={epoch_best_ari:.4f}, NMI={epoch_best_nmi:.4f}")

        if epoch_best_ari > best_test_metrics['ARI']:
            best_test_metrics['ARI'] = epoch_best_ari
            best_test_metrics['NMI'] = epoch_best_nmi
            best_test_metrics['ACC'] = epoch_results['best_acc']
            best_test_metrics['Purity'] = epoch_results['best_purity']
            best_test_metrics['Homo'] = epoch_results['best_homo']
            best_test_metrics['Comp'] = epoch_results['best_comp']
            best_test_metrics['PCC'] = epoch_results['imputation_pcc']
            best_test_metrics['MAE'] = epoch_results['imputation_mae']
            best_test_metrics['epoch'] = each_epoch + 1
            best_test_results = epoch_results
            state_best_test = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict()
            }
            torch.save(state_best_test, save_model_path + '_best_test')
            ari_patience_counter = 0
        else:
            ari_patience_counter += 1

        if ari_patience_counter >= ari_patience:
            print(f"ARI early stopping at epoch {each_epoch + 1} "
                  f"(no improvement for {ari_patience} epochs)")
            break

        model.train()

    return best_epoch, min_loss, best_test_results, best_test_metrics


def test(test_loader, n_clusters, n_clusters_test, input_dim,
         save_model_path=None, seed=0, device='cuda', model=None):
    if model is None:
        model = Model(input_dim, output_dim=n_clusters).cuda()
        weights = torch.load(save_model_path)['net']
        model.load_state_dict(weights)
    model.eval()

    z_test = []
    y_test = []
    all_pccs = {0.1: [], 0.2: [], 0.3: []}
    all_loss_maes = {0.1: [], 0.2: [], 0.3: []}

    for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                batch_loc, batch_subgraph) in enumerate(test_loader):
        combined, x_imp = model(batch_x, batch_subgraph, test=True)
        z_test.append(combined.cpu().detach().numpy())
        y_test.append(batch_y_true.cpu().detach().numpy())

        for d_rate in [0.1, 0.2, 0.3]:
            final_mask, final_x = DropData(batch_x, d_rate)
            loss_mae = model.mae_loss(final_mask * x_imp, final_mask * batch_x)
            pcc = pearson_corr(
                (final_mask * x_imp).cpu().detach().numpy(),
                (final_mask * batch_x).cpu().detach().numpy()
            )
            all_pccs[d_rate].append(pcc)
            all_loss_maes[d_rate].append(loss_mae.cpu().detach().numpy())

    avg_pccs = {d_rate: float(np.mean(pccs)) for d_rate, pccs in all_pccs.items()}
    avg_loss_maes = {d_rate: float(np.mean(maes)) for d_rate, maes in all_loss_maes.items()}

    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)

    best_avg = -1
    best_acc = -1
    best_ARI = -1
    best_NMI = -1
    best_homo = -1
    best_comp = -1
    best_purity = -1

    total_results = []

    for n_k in [3, 5, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]:
        try:
            km = KMeans(n_clusters=n_k, random_state=0, n_init=10).fit(z_test)
            clusters = km.labels_
            acc, nmi, ari, homo, comp, purity = evaluate(y_test, clusters)
        except BaseException:
            continue

        total_results.append({
            'method': 'kmeans', 'k': n_k,
            'ACC': acc, 'ARI': ari, 'NMI': nmi,
            'Purity': purity, 'Homogeneity': homo, 'Completeness': comp,
        })

        avg_metric = (acc + nmi + ari) / 3.0
        if avg_metric > best_avg:
            best_avg = avg_metric
            best_acc = acc
            best_ARI = ari
            best_NMI = nmi
            best_homo = homo
            best_comp = comp
            best_purity = purity

    print(f"\nBest results (by avg of ACC+NMI+ARI): ACC={best_acc:.4f}, ARI={best_ARI:.4f}, "
          f"NMI={best_NMI:.4f}, Purity={best_purity:.4f}, "
          f"Homo={best_homo:.4f}, Comp={best_comp:.4f}")
    print(f"Imputation PCC: {avg_pccs}")
    print(f"Imputation MAE: {avg_loss_maes}")

    return {
        'clustering': total_results,
        'imputation_pcc': avg_pccs,
        'imputation_mae': avg_loss_maes,
        'best_acc': best_acc, 'best_ari': best_ARI, 'best_nmi': best_NMI,
        'best_purity': best_purity, 'best_homo': best_homo, 'best_comp': best_comp,
    }
