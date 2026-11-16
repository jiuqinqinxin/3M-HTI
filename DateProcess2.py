import os
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from collections import defaultdict
from tqdm import tqdm
from iFeature.codes.PAAC import PAAC, readFasta, saveCode
from rdkit.Chem import AllChem, DataStructs

# 分子字符集映射（保持不变）
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}


class Smile2Morgan:
    # 保持原有实现不变
    def __init__(self, shape=1024, radius=2):
        self.shape = shape
        self.radius = radius

    def canonicalize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None

    def smiles_to_morgan(self, smile: str):
        smile = self.canonicalize(smile)
        if not smile:
            return np.zeros((self.shape,))
        mol = Chem.MolFromSmiles(smile)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.shape)
        features = np.zeros((self.shape,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features


class FeatureProcessor:
    def __init__(self, af2_npy_dir='AF2-npy'):
        self.atomDict = defaultdict(lambda: len(self.atomDict))
        self.fingerprintDict = defaultdict(lambda: len(self.fingerprintDict))
        self.bondDict = defaultdict(lambda: len(self.bondDict))
        self.edgeDict = defaultdict(lambda: len(self.edgeDict))
        self.wordDict = defaultdict(lambda: len(self.wordDict))
        self.morgan_featurizer = Smile2Morgan()
        self.af2_npy_dir = af2_npy_dir
        self.amino_acid_vocab = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }

    # 保持原有的分子处理方法不变
    def createAtoms(self, mol):
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atomDict[a] for a in atoms]
        return np.array(atoms)

    def createBondDict(self, mol):
        ijbondDict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bondDict[str(b.GetBondType())]
            ijbondDict[i].append((j, bond))
            ijbondDict[j].append((i, bond))

        for atom_idx in range(len(mol.GetAtoms())):
            if atom_idx not in ijbondDict:
                ijbondDict[atom_idx] = []
        return ijbondDict

    def getFingerprints(self, atoms, bondDict, radius):
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprintDict[a] for a in atoms]
        else:
            nodes = atoms
            ijedgeDict = bondDict

            for _ in range(radius):
                fingerprints = []
                for i, j_edge in ijedgeDict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprintDict[fingerprint])
                nodes = fingerprints

            ijedgeDict2 = defaultdict(lambda: [])
            for i, j_edge in self.edgeDict.items():
                for j, edge in j_edge:
                    bothSide = tuple(sorted((nodes[i], nodes[j])))
                    edge = self.edgeDict[(bothSide, edge)]
                    ijedgeDict2[i].append((j, edge))
            ijedgeDict = ijedgeDict2
        return np.array(fingerprints)

    def createAdjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)

    def splitSeqeunce(self, sequence, ngram=3):
        sequence = '-' + sequence + '='
        words = [self.wordDict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def process_smiles(self, smile):
        """处理单个SMILES字符串"""
        smile = smile.strip()
        if not smile:
            return None

        molecule_word = np.array([CHAR_SMI_SET[c] for c in smile if c in CHAR_SMI_SET])
        morgan_feature = self.morgan_featurizer.smiles_to_morgan(smile)

        mol = Chem.AddHs(Chem.MolFromSmiles(smile)) if Chem.MolFromSmiles(smile) else None
        if mol:
            atoms = self.createAtoms(mol)
            ijbondDict = self.createBondDict(mol)
            fingerprints = self.getFingerprints(atoms, ijbondDict, radius=2)
            adjacency = self.createAdjacency(mol)
        else:
            atoms = np.array([])
            fingerprints = np.array([])
            adjacency = np.array([])

        return {
            'molecule_word': molecule_word,
            'morgan_feature': morgan_feature,
            'atoms': atoms,
            'fingerprints': fingerprints,
            'adjacency': adjacency
        }

    def process_protein(self, sequence):
        """处理蛋白质序列"""
        words = self.splitSeqeunce(sequence)
        return words

    def build_protein_graph(self, sequence, uniprot_id, threshold=8.0):  # 移除 start_pos 和 end_pos 参数
        """
        使用AF2接触图构建蛋白质图（始终使用全序列）
        """
        sequence_length = len(sequence)
        effective_length = sequence_length  # 全序列长度即有效长度

        # 2. 独热编码节点特征（全序列）
        one_hot_features = self._get_amino_acid_one_hot(sequence)  # 直接使用完整序列

        # 3. 从AF2加载接触图并构建边（全序列）
        edge_index = self._get_edges_from_af2_contact_map(uniprot_id, sequence_length, threshold)  # 移除 start/end 参数

        return {
            'node_features': one_hot_features,
            'edge_index': edge_index,
            'sequence_length': effective_length
            # 移除 start_pos 和 end_pos 返回值
        }

    def _get_amino_acid_one_hot(self, sequence):
        """生成氨基酸独热编码"""
        num_amino_acids = len(self.amino_acid_vocab)
        one_hot_features = []

        for aa in sequence:
            if aa in self.amino_acid_vocab:
                one_hot = np.zeros(num_amino_acids)
                one_hot[self.amino_acid_vocab[aa]] = 1
                one_hot_features.append(one_hot)
            else:
                one_hot_features.append(np.zeros(num_amino_acids))

        return np.array(one_hot_features, dtype=np.float32)

    def _get_edges_from_af2_contact_map(self, uniprot_id, sequence_length, threshold):  # 移除 start/end 参数
        """从AF2接触图构建边（全序列）"""
        edge_index = []
        npy_path = os.path.join(self.af2_npy_dir, f"{uniprot_id}.npy")

        if not os.path.exists(npy_path):
            print(f"Warning: AF2 contact map not found for {uniprot_id}, using sequence proximity")
            return self._get_edges_from_sequence(sequence_length)  # 直接用全序列长度

        try:
            contact_map = np.load(npy_path)

            # 调整接触图形状以匹配序列长度（全序列）
            if contact_map.shape[0] != sequence_length:
                min_len = min(contact_map.shape[0], sequence_length)
                contact_map = contact_map[:min_len, :min_len]

            # 直接使用完整接触图（无需截取）
            n_residues = contact_map.shape[0]
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    if contact_map[i, j] <= threshold:
                        edge_index.append([i, j])
                        edge_index.append([j, i])

            return np.array(edge_index, dtype=np.int64) if edge_index else self._get_edges_from_sequence(
                sequence_length)

        except Exception as e:
            print(f"Error loading AF2 contact map for {uniprot_id}: {e}")
            return self._get_edges_from_sequence(sequence_length)

    def _get_edges_from_sequence(self, sequence_length, window_size=3):
        """备用：基于序列邻近性构建边"""
        edge_index = []
        for i in range(sequence_length):
            for j in range(max(0, i - window_size), min(sequence_length, i + window_size + 1)):
                if i != j:
                    edge_index.append([i, j])
        return np.array(edge_index, dtype=np.int64)


def process_csv(csv_path, smiles_col='SMILES', protein_col='Protein_Sequence',
                herb_id_col='Herb_ID', label_col='new_column', uniprot_col='UniProt',
                output_dir='features', lambda_value=10, af2_npy_dir='AF2-npy'):
    """
    处理CSV文件，提取特征并存储
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"读取CSV文件，共{len(df)}行数据")

    # 初始化处理器
    processor = FeatureProcessor(af2_npy_dir=af2_npy_dir)

    # --------------------------
    # 1. 提取并处理所有唯一的SMILES
    # --------------------------
    all_smiles = set()
    for smiles_str in df[smiles_col].astype(str):
        for smile in smiles_str.split(';'):
            smile = smile.strip()
            if smile:
                all_smiles.add(smile)
    all_smiles = list(all_smiles)
    print(f"共发现{len(all_smiles)}个唯一的SMILES")

    # 为SMILES分配编号并处理特征
    smiles_id_map = {smile: idx for idx, smile in enumerate(all_smiles)}
    smiles_features = []
    for smile in tqdm(all_smiles, desc="处理SMILES特征"):
        features = processor.process_smiles(smile)
        smiles_features.append(features)

    # --------------------------
    # 2. 提取并处理所有唯一的蛋白质
    # --------------------------
    # 创建蛋白质序列与起始/结束位置的映射
    protein_data = {}
    for _, row in df.iterrows():
        seq = row[protein_col]
        if seq not in protein_data:
            protein_data[seq] = {}

    protein_sequences = list(protein_data.keys())
    print(f"共发现{len(protein_sequences)}个唯一的蛋白质序列")

    # 为蛋白质分配编号
    protein_id_map = {seq: idx for idx, seq in enumerate(protein_sequences)}

    # 生成FASTA文件并计算PAAC特征
    fasta_file = os.path.join(output_dir, 'proteins.fasta')
    with open(fasta_file, "w") as f:
        for idx, seq in enumerate(protein_sequences):
            f.write(f'>Protein{idx}\n')
            f.write(f'{seq}\n')

    paac_output = os.path.join(output_dir, 'paac_features.tsv')
    if not os.path.exists(paac_output) or os.path.getsize(paac_output) <= 4096:
        print("计算PAAC特征...")
        fastas = readFasta.readFasta(fasta_file)
        encodings = PAAC(fastas, lambdaValue=lambda_value)
        saveCode.savetsv(encodings, paac_output)
    else:
        print("PAAC特征已存在，直接使用")

    # 读取PAAC特征并处理蛋白质特征
    paac_df = pd.read_csv(paac_output, sep='\t', header=None)

    # 创建蛋白质图字典和蛋白质特征列表
    protein_graph_dict = {}  # 新的蛋白质图字典
    protein_features = []  # 原有的蛋白质特征列表

    for idx, seq in enumerate(tqdm(protein_sequences, desc="处理蛋白质特征")):
        paac_feature = paac_df.iloc[idx + 1, 1:].values.astype(np.float32)
        protein_words = processor.process_protein(seq)

        # 获取对应的UniProt ID和起始/结束位置
        uniprot_id = df[df[protein_col] == seq][uniprot_col].iloc[0] if uniprot_col in df.columns else None

        # 构建蛋白质图（使用有效区域）
        graph_data = processor.build_protein_graph(seq, uniprot_id)

        # 将图数据保存到蛋白质图字典中（使用蛋白质编号作为键）
        protein_graph_dict[idx] = graph_data

        protein_features.append({
            'paac_feature': paac_feature,
            'words': protein_words
            # 注意：这里不再包含graph_data，因为图数据已单独保存在protein_graph_dict中
        })

    # --------------------------
    # 3. 构建标签和草药-成分映射
    # --------------------------
    labels_data = []
    herb_component_map = defaultdict(set)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建关联数据"):
        herb_id = row[herb_id_col]
        protein_seq = str(row[protein_col])
        protein_id = protein_id_map[protein_seq]
        label = int(row[label_col])

        labels_data.append({
            'herb_id': herb_id,
            'protein_id': protein_id,
            'label': label
        })

        smiles_str = str(row[smiles_col])
        for smile in smiles_str.split(';'):
            smile = smile.strip()
            if smile and smile in smiles_id_map:
                smiles_id = smiles_id_map[smile]
                herb_component_map[herb_id].add(smiles_id)

    # --------------------------
    # 4. 保存所有结果
    # --------------------------
    with open(os.path.join(output_dir, 'fingerprintDict.pickle'), 'wb') as f:
        pickle.dump(dict(processor.fingerprintDict), f)
    with open(os.path.join(output_dir, 'wordDict.pickle'), 'wb') as f:
        pickle.dump(dict(processor.wordDict), f)
    with open(os.path.join(output_dir, 'smiles_id_map.pickle'), 'wb') as f:
        pickle.dump(smiles_id_map, f)
    with open(os.path.join(output_dir, 'protein_id_map.pickle'), 'wb') as f:
        pickle.dump(protein_id_map, f)

    # 保存蛋白质图字典
    with open(os.path.join(output_dir, 'protein_graph_dict2.pickle'), 'wb') as f:
        pickle.dump(protein_graph_dict, f)

    np.save(os.path.join(output_dir, 'smiles_features.npy'),
            np.array(smiles_features, dtype=object), allow_pickle=True)
    np.save(os.path.join(output_dir, 'protein_features.npy'),
            np.array(protein_features, dtype=object), allow_pickle=True)

    np.save(os.path.join(output_dir, 'labels.npy'),
            np.array(labels_data, dtype=object), allow_pickle=True)

    herb_component_dict = {k: list(v) for k, v in herb_component_map.items()}
    with open(os.path.join(output_dir, 'herb_component_map.pickle'), 'wb') as f:
        pickle.dump(herb_component_dict, f)

    print(f"特征提取完成，保存至{output_dir}目录")
    print(f"已保存{len(smiles_features)}个SMILES特征和{len(protein_features)}个蛋白质特征")
    print(f"已保存{len(protein_graph_dict)}个蛋白质图")


if __name__ == '__main__':
    process_csv(
        csv_path='cold_start_data/test.csv',
        smiles_col='SMILES',
        protein_col='Protein_Sequence',
        herb_id_col='Herb_ID',
        label_col='new_column',
        uniprot_col='UniProt',
        output_dir='cold_start_data/test',
        af2_npy_dir='AF2-npy'
    )