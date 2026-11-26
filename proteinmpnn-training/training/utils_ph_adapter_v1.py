import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os


def _as_scalar(value, default):
    """Convert scalar tensors/arrays from torch serialization to Python."""
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return default
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return default
        return value.reshape(-1)[0].item()
    return value

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYcdhekrX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )




def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.where(t['idx']==idx)[0]
                        initial_sequence = "".join(np.array(list(t['seq']))[res])
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[10:]
                        if res.shape[0] < 4:
                            pass
                        else:
                            my_dict['seq_chain_'+letter] = "".join(np.array(list(t['seq']))[res])
                            concat_seq += my_dict['seq_chain_'+letter]
                            ##
                            res_indices = np.where(t['idx'] == idx)[0]
                            if len(res_indices) > 0 and t['masked'][res_indices[0]] == 1:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
			    ##
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])  #[L, 14, 3]
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    my_dict['crystal_pH'] = float(
                        _as_scalar(t.get('crystal_pH'), float('nan'))
                    )
                    my_dict['ph_from_cif'] = bool(
                        _as_scalar(t.get('ph_from_cif'), False)
                    )
                    if (
                        len(concat_seq) >= 4
                        and my_dict['num_of_chains'] > 0
                        and len(concat_seq) <= max_length
                        and my_dict['ph_from_cif']
                        and np.isfinite(my_dict['crystal_pH'])
                        and 0.0 <= my_dict['crystal_pH'] <= 14.0
                    ):
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out



def loader_pdb(item, params):
    """
    Загружает одну цепь вместе со всеми цепями её биологической сборки.
    Применяет assembly трансформации (rotation + translation) к xyz.

    item[0] = "{pdbid}_{chid}", например "1abc_A" или "9h5k_BA"
    params  = {'DIR': '/data/pdb_proton', 'MASK': 0.15, ...}

    Формат вывода совместим с оригинальным ProteinMPNN featurize().
    """
    import random

    pdbid, chid = item[0].split('_', 1)  # split('_', 1) для двухбуквенных chain ID
    PREFIX = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)

    # Проверяем наличие metadata и цепи
    if not os.path.isfile(PREFIX + ".pt"):
        return {'seq': np.zeros(5)}

    try:
        meta = torch.load(PREFIX + ".pt", weights_only=False, map_location='cpu')
    except Exception:
        return {'seq': np.zeros(5)}

    asmb_ids    = meta.get('asmb_ids',    ['1'])
    asmb_chains = meta.get('asmb_chains', [chid])

    # Находим все assembly в которых участвует наша цепь
    asmb_candidates = list({
        (a, c) for a, c in zip(asmb_ids, asmb_chains)
        if chid in c.split(',')
    })

    if not asmb_candidates:
        # Цепь не нашлась ни в одной assembly — грузим одну цепь без контекста
        asmb_candidates = [('1', chid)]

    # Случайно выбираем одну из сборок (data augmentation)
    asmb_i, _ = random.choice(asmb_candidates)
    xform_key  = f'asmb_xform{asmb_i}'

    # Собираем все записи для выбранной assembly
    idx = [k for k, a in enumerate(asmb_ids) if a == asmb_i]

    # Уникальные цепи в этой assembly
    unique_chains = list({c for k in idx for c in asmb_chains[k].split(',')})

    # Загружаем все цепи сборки
    chain_data = {}
    for c in unique_chains:
        chain_path = "%s_%s.pt" % (PREFIX, c)
        if os.path.isfile(chain_path):
            try:
                chain_data[c] = torch.load(chain_path,
                    weights_only=False, map_location='cpu')
            except Exception:
                pass

    if chid not in chain_data:
        return {'seq': np.zeros(5)}

    # Применяем трансформации к координатам
    xforms = meta.get(xform_key)  # np.array (N_ops, 4, 4) или None

    for k in idx:
        chains_in_op = asmb_chains[k].split(',')
        for c in chains_in_op:
            if c not in chain_data:
                continue
            if xforms is None or k >= len(xforms):
                continue
            M = xforms[k] if isinstance(xforms, np.ndarray) else np.array(xforms[k])
            xyz = chain_data[c]['xyz']   # (L, 14, 3)
            if isinstance(xyz, torch.Tensor):
                xyz = xyz.numpy()
            # Применяем 4×4 матрицу: [R|t] к однородным координатам
            shape = xyz.shape                            # (L, 14, 3)
            xyz_flat = xyz.reshape(-1, 3)               # (L*14, 3)
            ones = np.ones((xyz_flat.shape[0], 1), dtype=np.float32)
            xyz_h = np.concatenate([xyz_flat, ones], axis=-1)  # (L*14, 4)
            xyz_t = (M @ xyz_h.T).T[:, :3]             # (L*14, 3)
            chain_data[c]['xyz'] = torch.from_numpy(
                xyz_t.reshape(shape).astype(np.float32))

    # Определяем какие цепи маскировать (предсказывать) vs контекст
    # Маскируем только целевую цепь; остальные служат структурным контекстом
    seq_parts    = []
    xyz_parts    = []
    idx_parts    = []
    masked_parts = []
    offset = 0

    # Сначала целевая цепь
    target = chain_data[chid]
    xyz_t = target['xyz']
    xyz_t_np = xyz_t.numpy() if isinstance(xyz_t, torch.Tensor) else np.array(xyz_t)
    # Обрезаем до наблюдаемых остатков (CA не NaN) — как в оригинальном ProteinMPNN
    observed_t = np.where(~np.isnan(xyz_t_np[:, 1, :]).any(axis=1))[0]
    if len(observed_t) < 4:
        return {'seq': np.zeros(5)}
    seq_t = ''.join(target['seq'][i] for i in observed_t)
    xyz_t_np = xyz_t_np[observed_t].astype(np.float32)
    L_t = len(seq_t)
    seq_parts.append(seq_t)
    xyz_parts.append(torch.from_numpy(xyz_t_np))
    chain_id = 0
    idx_parts.append(torch.full((L_t,), chain_id, dtype=torch.int32))
    masked_parts.append(torch.ones(L_t))   # 1 = предсказываем
    offset = L_t + 200

    # Потом остальные цепи сборки как контекст
    for c in unique_chains:
        if c == chid or c not in chain_data:
            continue
        ctx = chain_data[c]
        L_c = len(ctx['seq'])
        if L_c < 4:
            continue
        xyz_c = ctx['xyz']
        xyz_c_np = xyz_c.numpy() if isinstance(xyz_c, torch.Tensor) else np.array(xyz_c)
        # Обрезаем до наблюдаемых остатков
        observed_c = np.where(~np.isnan(xyz_c_np[:, 1, :]).any(axis=1))[0]
        if len(observed_c) < 4:
            continue
        seq_c = ''.join(ctx['seq'][i] for i in observed_c)
        xyz_c_np = xyz_c_np[observed_c].astype(np.float32)
        L_c = len(seq_c)
        seq_parts.append(seq_c)
        xyz_parts.append(torch.from_numpy(xyz_c_np))
        idx_parts.append(torch.arange(offset, offset + L_c))
        masked_parts.append(torch.zeros(L_c))   # 0 = контекст, не предсказываем
        offset += L_c + 200

    seq_cat    = ''.join(seq_parts)
    xyz_cat    = torch.cat(xyz_parts, dim=0)
    idx_cat    = torch.cat(idx_parts).int()
    masked_cat = torch.cat(masked_parts).int()

    if len(seq_cat) < 4:
        return {'seq': np.zeros(5)}

    return {
        'seq':    seq_cat,
        'xyz':    xyz_cat,
        'idx':    idx_cat,
        'masked': masked_cat,
        'label':  item[0],
        'crystal_pH': float(_as_scalar(meta.get('crystal_pH'), float('nan'))),
        'ph_from_cif': bool(_as_scalar(meta.get('ph_from_cif'), False)),
    }




def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test
