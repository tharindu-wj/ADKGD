import numpy as np
import os
import random
import torch
import math
from random import shuffle


class Reader:
    def __init__(self, args, path):
        # Stored so get_data() can consult args.neg_source / args.gan_path
        # (path to kggan .pt checkpoint when neg_source='gan') without changing
        # its signature. Optional flags use getattr() below.
        self.args = args

        self.ent2id = dict()
        self.rel2id = dict()
        self.id2ent = dict()
        self.id2rel = dict()
        # 一个字典，头实体到尾实体的映射，表示从一个实体（头部）可以到达的所有实体（尾部）
        self.h2t = {}
        # 一个字典，尾实体到头实体的映射，表示可以到达一个实体（尾部）的所有实体（头部）
        self.t2h = {}

        # 记录异常数据的数量
        self.num_anomalies = 0
        # 存储所有的三元组（实体-关系-实体）
        self.triples = []
        # 用于批处理的起始索引
        self.start_batch = 0
        self.path = path

        self.A = {}
        #读取所有的数据，从train.txt， valid.txt, test.txt
        self.read_triples()
        # 存储原始三元组的集合，用于快速检查是否存在某个特定的三元组
        self.triple_ori_set = set(self.triples)
        # 记录原始三元组的数量
        self.num_original_triples = len(self.triples)
        # 初始数据的entity总数，relation总数
        self.num_entity = self.num_ent()
        self.num_relation = self.num_rel()
        print('entity&relation: ', self.num_entity, self.num_relation)
        # 生成异常数据
        self.bp_triples_label = self.inject_anomaly(args)
        # 正常数据+异常数据，混合，含标签信息
        self.num_triples_with_anomalies = len(self.bp_triples_label)
        # 用于生成更多的异常数据
        self.train_data, self.labels = self.get_data()
        # 等同于self.bp_triples_label
        self.triples_with_anomalies, self.triples_with_anomalies_labels = self.get_data_test()

    # def train_triples(self):
    #     return self.triples["train"]
    #
    # def valid_triples(self):
    #     return self.triples["valid"]
    #
    # def test_triples(self):
    #     return self.triples["test"]

    # def all_triples(self):
    #     return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open(entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open(relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def read_triples(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r") as f:
                for line in f.readlines():
                    try:
                        head, rel, tail = line.strip().split("\t")
                    except:
                        print(line)
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

        print("Read end!")
        return self.triples

    def rand_ent_except(self, ent):
        rand_ent = random.randint(1, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def generate_anomalous_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 2)
            if head_or_tail == 0:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = rel
                new_tail = tail
                # neg_triples.append((new_head, rel, tail))
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = tail
            else:
                # new_tail = self.rand_ent_except(tail)
                # neg_triples.append((head, rel, new_tail))
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, self.num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
            while anomaly in self.triple_ori_set:
                if head_or_tail == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                    new_relation = rel
                    new_tail = tail
                    # neg_triples.append((new_head, rel, tail))
                elif head_or_tail == 1:
                    new_head = head
                    new_relation = random.randint(0, self.num_relation - 1)
                    new_tail = tail
                else:
                    # new_tail = self.rand_ent_except(tail)
                    # neg_triples.append((head, rel, new_tail))
                    new_head = head
                    new_relation = rel
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            neg_triples.append(anomaly)
        return neg_triples

    def generate_anomalous_triples_2(self, num_anomaly):
        neg_triples = []
        for i in range(num_anomaly):
            new_head = random.randint(0, self.num_entity - 1)
            new_relation = random.randint(0, self.num_relation - 1)
            new_tail = random.randint(0, self.num_entity - 1)

            anomaly = (new_head, new_relation, new_tail)

            while anomaly in self.triple_ori_set:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)

            neg_triples.append(anomaly)
        return neg_triples

    def shred_triples(self, triples):
        h_dix = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_dix, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    # def all_triplets(self):
    #     ph_all, pr_all, pt_all = self.shred_triples(self.triples)
    #     nh_all, nr_all, nt_all = self.shred_triples(self.generate_neg_triples(self.triples))
    #     return ph_all, pt_all, nh_all, nt_all, pr_all

    def get_data(self):
        # bp_triples_label = self.inject_anomaly()
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        # Phase B: source of training-time negatives (set C).
        # 'gan'    -> in-process call to kggan's generator (loaded once from a
        #             .pt checkpoint at --gan_path). Same masked-decode + retry +
        #             uniform-random-fallback logic as kggan's TSV exporter, but
        #             returned directly instead of routed through a file.
        #             Used uniformly for both real positives AND injected eval
        #             anomalies in bp_triples_label.
        # 'random' -> ADKGD's original per-positive random corruption (default).
        neg_source = getattr(self.args, 'neg_source', 'random')
        if neg_source == 'gan':
            bn_triples = self._gan_negatives(bp_triples)
        elif neg_source == 'lp_band':
            bn_triples = self._lp_negatives(bp_triples)
        else:
            bn_triples = self.generate_anomalous_triples(bp_triples)

        # 前一半是正常数据，后一半是异常数据
        all_triples = bp_triples + bn_triples

        return self.toarray(all_triples), self.toarray(labels)

    def _gan_negatives(self, pos_triples, replace_nulls=True):
        """Generate one negative per positive by running the GAN in-process.

        Treats every entry in `pos_triples` uniformly -- real positives AND
        injected eval anomalies. The generator picks a head/tail slot by
        corruptibility, decodes under type-pool + known-true + self masks,
        and redraws up to a bound; rows that STILL fail come back as null
        corruptions (the original triple), flagged in stats['null_indices'].

        A null is a real fact: training on it as a 'negative' injects label
        noise, so for the TRAINING role (replace_nulls=True) null rows are
        replaced with ADKGD's own random corruption, counted and logged. The
        eval role (inject_anomaly) passes replace_nulls=False and filters
        nulls itself via the genuine-corruption check.
        """
        if not hasattr(self, '_gan_payload') or self._gan_payload is None:
            self._load_gan_model()

        from kgsage_bridge.bridge import generate, render_stats

        negatives, stats = generate(
            pos_triples,
            payload=self._gan_payload,
            adkgd_id2ent=self.id2ent,
            adkgd_id2rel=self.id2rel,
            adkgd_ent2id=self.ent2id,
            adkgd_rel2id=self.rel2id,
            rng=self._gan_rng,
        )
        print('[GAN] ' + render_stats(stats))
        null_idx = stats.get('null_indices', [])
        if replace_nulls and null_idx:
            fillers = self.generate_anomalous_triples(
                [pos_triples[i] for i in null_idx])
            for j, i in enumerate(null_idx):
                negatives[i] = fillers[j]
            print('[GAN] %d null corruptions replaced with random fallbacks '
                  'for the training role' % len(null_idx))
        self._print_pair_preview('GAN', pos_triples, negatives)
        return negatives

    def _lp_negatives(self, pos_triples):
        """Option B: one close-but-false negative per positive from the frozen
        LP band sampler (type-valid, all-splits-masked, top-k band below
        s(true)). No null corruptions by construction -- the sampler's
        fallback ladder always yields a genuine single-slot corruption."""
        if not hasattr(self, '_lp_payload') or self._lp_payload is None:
            self._load_lp_sampler()

        from kgsage_bridge.bridge import generate_band, render_band_stats

        negatives, stats = generate_band(
            pos_triples,
            payload=self._lp_payload,
            adkgd_id2ent=self.id2ent,
            adkgd_id2rel=self.id2rel,
            adkgd_ent2id=self.ent2id,
            adkgd_rel2id=self.rel2id,
            rng=self._lp_rng,
        )
        print(render_band_stats(stats))
        self._print_pair_preview('lp_band', pos_triples, negatives)
        return negatives

    def _load_lp_sampler(self):
        """Build the band sampler once, cache on self. Missing --lp_path /
        --lp_ids_dir fails loudly, mirroring the GAN checkpoint policy."""
        import sys as _sys
        from pathlib import Path as _Path

        _experiments_dir = _Path(__file__).resolve().parent / 'experiments'
        if str(_experiments_dir) not in _sys.path:
            _sys.path.insert(0, str(_experiments_dir))

        from kgsage_bridge.bridge import load_lp
        import numpy as _np

        lp_path = getattr(self.args, 'lp_path', None)
        lp_ids_dir = getattr(self.args, 'lp_ids_dir', None)
        if not lp_path or not lp_ids_dir:
            raise ValueError(
                "--lp_path and --lp_ids_dir are required for 'lp_band' "
                "(fetch them with: python -m kgsage.cli.fetch_lp)")
        self._lp_payload = load_lp(
            lp_path, lp_ids_dir, self.args.data_path,
            band_k=getattr(self.args, 'band_k', 10),
            band_temp=getattr(self.args, 'band_temp', 0.5))
        self._lp_rng = _np.random.default_rng(getattr(self.args, 'seed', 0))
        print('[lp_band] sampler ready (ckpt=%s, masks from %s)'
              % (lp_path, self.args.data_path))

    def _print_pair_preview(self, tag, pos_triples, negatives):
        """Log a capped preview of (positive -> negative) pairs."""
        _preview = int(os.environ.get('GAN_PAIR_PREVIEW', '20'))
        n = len(pos_triples)
        print('[%s] %d (positive -> negative) pairs (showing first %d):'
              % (tag, n, min(_preview, n)))
        for i in range(min(_preview, n)):
            ph, pr, pt = pos_triples[i]
            nh, nr, nt = negatives[i]
            moved = []
            if ph != nh:
                moved.append('head')
            if pr != nr:
                moved.append('relation')
            if pt != nt:
                moved.append('tail')
            moved_str = ','.join(moved) if moved else 'NONE'
            print('  pos: (%s, %s, %s)'
                  % (self.id2ent[ph], self.id2rel[pr], self.id2ent[pt]))
            print('  neg: (%s, %s, %s)  [moved: %s]'
                  % (self.id2ent[nh], self.id2rel[nr], self.id2ent[nt], moved_str))
        if n > _preview:
            print('  ... (%d more pairs suppressed; set GAN_PAIR_PREVIEW to raise)'
                  % (n - _preview))

    def _load_gan_model(self):
        """Load the GAN checkpoint once, cache on self.

        The checkpoint bundles the generator weights + the GAN's vocab maps +
        the set of real triples, so we don't need to rebuild the KG here.

        FileNotFoundError on a bad path is intentionally NOT caught -- when
        --neg_source=gan is requested, a missing checkpoint should fail loudly.
        """
        import sys as _sys
        from pathlib import Path as _Path

        # Put experiments/ on sys.path so `from kgsage_bridge.bridge import ...` works.
        # See experiments/README.md for the bridge architecture rationale.
        _experiments_dir = _Path(__file__).resolve().parent / 'experiments'
        if str(_experiments_dir) not in _sys.path:
            _sys.path.insert(0, str(_experiments_dir))

        from kgsage_bridge.bridge import load_gan
        import numpy as _np

        ckpt_path = getattr(self.args, 'gan_path',
                            'experiments/kgsage/outputs/checkpoints/dummy.pt')
        self._gan_payload = load_gan(ckpt_path)
        seed = getattr(self.args, 'seed', 0)
        self._gan_rng = _np.random.default_rng(seed)
        print('[GAN] loaded checkpoint from %s (device=%s)'
              % (ckpt_path, self._gan_payload['device']))

    def get_data_test(self):
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        return self.toarray(bp_triples), self.toarray(labels)

    def toarray(self, x):
        return torch.from_numpy(np.array(list(x)).astype(np.int32))

    def inject_anomaly(self, args):
        print("Inject anomalies!")
        original_triples = self.triples
        triple_size = len(original_triples)

        # 计算注入的异常数量
        self.num_anomalies = int(args.anomaly_ratio * self.num_original_triples)
        args.num_anomaly_num = self.num_anomalies
        print("###########Inject TOP@K% Anomalies##########")
        # if self.isInjectTopK:
        #     self.num_anomalies = args.num_anomaly_num
        #     print("###########Inject TOP@K Anomalies##########")
        # else:
        #

        # Source of the INJECTED eval anomalies (the label-1 triples ADKGD detects,
        # and -- because ADKGD is transductive -- the anomalies polluting the graph).
        #   'random' -> ADKGD original: half single-slot corruption of real triples,
        #               half fully-random triples.
        #   'gan'    -> KGSAGE corruptions. We OVERSAMPLE and keep only GENUINE
        #               corruptions (differ from their source) so no real triple is
        #               mislabelled as an anomaly -- the single-shot generator keeps
        #               the original triple on a self-loop/collision (used_original).
        test_source = getattr(args, 'test_anomaly_source', 'random')
        if test_source in ('gan', 'lp_band'):
            over = min(self.num_original_triples, int(self.num_anomalies * 1.5) + 1)
            idx = random.sample(range(0, self.num_original_triples), over)
            selected_triples = [original_triples[i] for i in idx]
            if test_source == 'gan':
                # eval role: nulls are filtered below by the genuine-corruption
                # check, so no random replacement (would blur attribution)
                corrupted = self._gan_negatives(selected_triples, replace_nulls=False)
            else:
                corrupted = self._lp_negatives(selected_triples)
            anomalies = [c for src, c in zip(selected_triples, corrupted)
                         if tuple(c) != tuple(src)][:self.num_anomalies]
            if len(anomalies) < self.num_anomalies:
                print('[test-anomaly %s] only %d/%d genuine anomalies '
                      '(generator collided on the rest)'
                      % (test_source, len(anomalies), self.num_anomalies))
        else:
            # 随机选择一半的异常数量对应的索引，从原始三元组中生成第一部分异常数据
            idx = random.sample(range(0, self.num_original_triples - 1), self.num_anomalies // 2)
            selected_triples = [original_triples[idx[i]] for i in range(len(idx))]
            # anomalies1：用已有的entities替换；anomalies2：从整个空间随机生成另一半
            anomalies = self.generate_anomalous_triples(selected_triples) \
                        + self.generate_anomalous_triples_2(self.num_anomalies // 2)

        # B0 hygiene: the realised anomaly count can be smaller than requested
        # (gan-branch shortfall after the genuine-corruption filter; random
        # branch's //2 rounding). test() uses num_anomalies as the recall
        # denominator and max_top_k, so keep it in sync with reality.
        if len(anomalies) != self.num_anomalies:
            print('[inject_anomaly] realised %d anomalies (requested %d) -- '
                  'num_anomalies updated' % (len(anomalies), self.num_anomalies))
        self.num_anomalies = len(anomalies)
        args.num_anomaly_num = self.num_anomalies

        triple_label = [(original_triples[i], 0) for i in range(len(original_triples))]
        anomaly_label = [(anomalies[i], 1) for i in range(len(anomalies))]
        # 将带有标签的原始三元组和异常三元组合并成一个列表。
        triple_anomaly_label = triple_label + anomaly_label
        # 使用 shuffle 函数将这个列表打乱，以确保数据的随机性。
        shuffle(triple_anomaly_label)
        return triple_anomaly_label

#
# dataset = Reader(args.data_dir_FB, "train")
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
# print(xxx[0])
