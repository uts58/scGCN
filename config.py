import glob
import os
import pickle
from collections import OrderedDict

import pandas as pd
from torch_geometric.data import Batch

config_ = {
    'config_name': 'uts et. al',
    'parent_dir': "/mmfs1/scratch/utsha.saha/mouse_data/data/",
    'pairs_data_dir': "/mmfs1/scratch/utsha.saha/mouse_data/data/pairs/",
    'graph_dir': '/mmfs1/scratch/utsha.saha/mouse_data/data/graphs/brain',
    'chrom_list': [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
        'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
        'chr16', 'chr17', 'chr18', 'chr19',
        # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
        'chrX'
    ],
    'rna_umicount_list': [
        "/mmfs1/scratch/utsha.saha/mouse_data/data/rna/GSE223917_HiRES_brain.rna.umicount.tsv",
        "/mmfs1/scratch/utsha.saha/mouse_data/data/rna/GSE223917_HiRES_emb.rna.umicount.tsv"
    ],
    'chr_gene_mapping_gencode': "/mmfs1/scratch/utsha.saha/mouse_data/data/chr_gene_mapping/gencode.vM25.annotation.gtf",
    'chr_gene_mapping_ncbi': "/mmfs1/scratch/utsha.saha/mouse_data/data/chr_gene_mapping/ncbi_grmc38_p6_annotation.gtf",
    'labels': "/mmfs1/scratch/utsha.saha/mouse_data/data/labels_brain.csv",
    'resolution': 50000
}

resolution = config_['resolution']


def create_directory(directory_list: list):
    for directory_name in directory_list:
        os.makedirs(directory_name, exist_ok=True)


create_directory([config_['pairs_data_dir'], config_['graph_dir']])


def create_chr_gene_mapping() -> dict:
    def process_csv(file_path):
        df = pd.read_csv(file_path,
                         sep='\t',
                         comment='#',
                         names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame',
                                'attribute'])
        df = df.loc[df['feature'] == 'gene'].copy()
        return df

    df_chr_gene_mapping_ncbi = process_csv(config_['chr_gene_mapping_ncbi'])

    df_chr_gene_mapping_ncbi['gene_name'] = df_chr_gene_mapping_ncbi['attribute'].str.extract(r'gene "([^"]+)"')
    df_chr_gene_mapping_ncbi['gene_synonyms'] = df_chr_gene_mapping_ncbi['attribute'].str.findall(
        r'gene_synonym "([^"]+)"')
    df_chr_gene_mapping_ncbi = df_chr_gene_mapping_ncbi[['gene_name', 'gene_synonyms']]

    df_chr_gene_mapping = process_csv(config_['chr_gene_mapping_gencode'])
    df_chr_gene_mapping['gene_name'] = df_chr_gene_mapping['attribute'].str.extract(r'gene_name "([^"]+)"')
    df_chr_gene_mapping = df_chr_gene_mapping[['seqname', 'start', 'end', 'gene_name']]
    df_chr_gene_mapping = pd.merge(df_chr_gene_mapping, df_chr_gene_mapping_ncbi, on='gene_name', how='left')

    def process_row(row):
        if type(row['gene_synonyms']) == list:
            x = set(row['gene_synonyms'] + [row['gene_name']])
            return list(x)
        else:
            return [row['gene_name']]

    df_chr_gene_mapping['gene_synonyms'] = df_chr_gene_mapping.apply(process_row, axis=1)
    df_chr_gene_mapping = df_chr_gene_mapping.loc[df_chr_gene_mapping['seqname'].isin(config_['chrom_list'])].copy()

    data_ = {}
    for i, rows in df_chr_gene_mapping.iterrows():
        for items in rows['gene_synonyms']:
            data_[items] = {
                'original_gene_name': rows['gene_name'],
                'seqname': rows['seqname'],
                'start': rows['start'],
                'end': rows['end'],
                'bin': (rows['start'] + rows['end']) // 2 // resolution
            }

    return data_


def get_data_files():
    cell_vs_files = {}

    for i in glob.glob(f"{config_['pairs_data_dir']}/*.pairs"):
        file_name = i.replace('.pairs', '')
        cell_name = file_name.split('_')[-1]
        cell_vs_files[cell_name] = file_name

    return cell_vs_files


def load_graph_data() -> dict:
    graph_dict = OrderedDict()
    for files in glob.glob(f'{config_["graph_dir"]}/*.pkl'):
        graph_data = pickle.load(open(files, 'rb'))
        graph_data.x, graph_data.edge_index = graph_data.x.float(), graph_data.edge_index.long()
        name = files.split('/')[-1].replace('.pkl', '')
        graph_dict[name] = graph_data

    return graph_dict


def chunk_graphs(graph_list, batch_size):
    """Chunks a list of graphs into batches of specified size."""

    for i in range(0, len(graph_list), batch_size):
        batch = graph_list[i:i + batch_size]
        batch = Batch.from_data_list(batch)  # Combine into a Batch object
        yield batch
